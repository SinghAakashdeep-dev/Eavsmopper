"""
The overlay window: small, always on top, draggable, and (on Windows 10 2004+) invisible
to screen sharing / recording, so only you see it.

Threads:
  - LiveTranscriber (audio + Whisper) calls back with each new line
  - each answer is generated in its own thread and streamed into the window
Qt widgets may only be touched from the main thread, so worker threads talk to the
window through Qt Signals (the Bridge class). Signals are thread-safe.
"""

import ctypes
import sys
import threading
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFrame, QHBoxLayout, QLabel, QPlainTextEdit,
    QPushButton, QSizeGrip, QSplitter, QTextBrowser, QVBoxLayout, QWidget,
)

from backend.assistant import Assistant, looks_like_question
from backend.knowledge import KnowledgeBase
from backend.router import PROVIDERS, get_provider
from backend.settings import save_settings
from backend.transcriber import LiveTranscriber

SOURCES = {"system": "Call audio", "mic": "Microphone", "both": "Call + mic"}
DEVICES = {"auto": "Auto", "cpu": "CPU", "gpu": "GPU (NVIDIA)"}
WHISPER_SPEEDS = {"tiny.en": "Fastest", "base.en": "Fast", "small.en": "Accurate"}

WDA_NONE = 0x0
WDA_EXCLUDEFROMCAPTURE = 0x11   # Windows 10 2004+: window is skipped by screen capture


def set_hidden_from_capture(widget, hidden):
    if sys.platform != "win32":
        return False
    hwnd = int(widget.winId())
    flag = WDA_EXCLUDEFROMCAPTURE if hidden else WDA_NONE
    return bool(ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, flag))


STYLE = """
#card { background: rgba(18, 20, 26, 232); border: 1px solid rgba(255,255,255,40); border-radius: 12px; }
QLabel, QCheckBox { color: #e6e6e6; font-size: 12px; }
#title { font-weight: 600; font-size: 13px; }
#status { color: #9aa4b2; }
QPushButton { background: #2b303b; color: #e6e6e6; border: none; border-radius: 6px; padding: 4px 10px; }
QPushButton:hover { background: #394050; }
#primary { background: #3b6df0; } #primary:hover { background: #4b7cff; }
QComboBox { background: #2b303b; color: #e6e6e6; border: none; border-radius: 6px; padding: 3px 8px; }
QTextBrowser, QPlainTextEdit { background: rgba(0,0,0,70); color: #f0f0f0; border: none;
                               border-radius: 8px; font-size: 13px; padding: 6px; }
"""


class Bridge(QObject):
    transcript = Signal(str, str)
    status = Signal(str)
    partial = Signal(str, str)
    answer_start = Signal(str)
    answer_chunk = Signal(str)
    answer_done = Signal()


class DragBar(QFrame):
    """Header you can drag to move the frameless window."""

    def mousePressEvent(self, e):
        self._offset = e.globalPosition().toPoint() - self.window().frameGeometry().topLeft()

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.LeftButton:
            self.window().move(e.globalPosition().toPoint() - self._offset)


class Overlay(QWidget):
    def __init__(self, root: Path, settings: dict, settings_path: Path):
        super().__init__()
        self.root, self.settings, self.settings_path = root, settings, settings_path
        self.kb = KnowledgeBase()
        self.assistant = Assistant(self.kb)
        self.transcriber = None
        self.answering = False
        self.pending_question = None
        self.partials = {}          # speaker -> words heard so far in the current phrase
        # Typewriter: answer text goes into a buffer and is revealed a few characters every 15 ms
        self._type_buffer = ""
        self._answer_finished = False

        self.bridge = Bridge()
        self.bridge.transcript.connect(self._on_transcript)
        self.bridge.status.connect(self._set_status)
        self.bridge.partial.connect(self._on_partial)
        self.bridge.answer_start.connect(self._on_answer_start)
        self.bridge.answer_chunk.connect(self._on_answer_chunk)
        self.bridge.answer_done.connect(self._on_answer_done)

        self.setWindowTitle("Eavsmopper")
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet(STYLE)
        self.resize(440, 560)
        self._build_ui()
        self._type_timer = QTimer(self)
        self._type_timer.timeout.connect(self._type_tick)
        self._type_timer.start(15)
        self._reindex()

    # ---------------- UI ----------------
    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        card = QFrame(objectName="card")
        outer.addWidget(card)
        lay = QVBoxLayout(card)
        lay.setContentsMargins(10, 8, 10, 8)

        # header (drag here)
        bar = DragBar()
        h = QHBoxLayout(bar)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(QLabel("Eavsmopper", objectName="title"))
        self.status_lbl = QLabel("Idle", objectName="status")
        h.addWidget(self.status_lbl, 1)
        self.start_btn = QPushButton("Start", objectName="primary")
        self.start_btn.clicked.connect(self.toggle_listening)
        h.addWidget(self.start_btn)
        ask_btn = QPushButton("Answer")
        ask_btn.setToolTip("Answer the latest question in the transcript")
        ask_btn.clicked.connect(self.answer_latest)
        h.addWidget(ask_btn)
        gear = QPushButton("⚙")
        gear.clicked.connect(lambda: self.settings_box.setVisible(not self.settings_box.isVisible()))
        h.addWidget(gear)
        close = QPushButton("✕")
        close.clicked.connect(self.close)
        h.addWidget(close)
        lay.addWidget(bar)

        # settings panel
        self.settings_box = QFrame()
        s = QVBoxLayout(self.settings_box)
        s.setContentsMargins(0, 4, 0, 4)
        row1 = QHBoxLayout()
        self.source_cb = QComboBox()
        for key, label in SOURCES.items():
            self.source_cb.addItem(label, key)
        self.source_cb.setCurrentIndex(list(SOURCES).index(self.settings["source"]))
        self.source_cb.currentIndexChanged.connect(self._change_source)
        row1.addWidget(QLabel("Listen to"))
        row1.addWidget(self.source_cb, 1)
        self.provider_cb = QComboBox()
        for key, label in PROVIDERS.items():
            self.provider_cb.addItem(label, key)
        self.provider_cb.setCurrentIndex(list(PROVIDERS).index(self.settings["provider"]))
        self.provider_cb.currentIndexChanged.connect(lambda: self._save("provider", self.provider_cb.currentData()))
        row1.addWidget(QLabel("AI"))
        row1.addWidget(self.provider_cb, 1)
        s.addLayout(row1)

        row2 = QHBoxLayout()
        self.auto_chk = QCheckBox("Auto-answer questions")
        self.auto_chk.setChecked(self.settings["auto_answer"])
        self.auto_chk.toggled.connect(lambda v: self._save("auto_answer", v))
        row2.addWidget(self.auto_chk)
        self.hide_chk = QCheckBox("Hide from screen share")
        self.hide_chk.setChecked(self.settings["hide_from_share"])
        self.hide_chk.toggled.connect(self._toggle_hidden)
        row2.addWidget(self.hide_chk)
        s.addLayout(row2)

        row_speed = QHBoxLayout()
        row_speed.addWidget(QLabel("Transcription"))
        self.speed_cb = QComboBox()
        for key, label in WHISPER_SPEEDS.items():
            self.speed_cb.addItem(f"{label} ({key})", key)
        if self.settings["whisper_model"] in WHISPER_SPEEDS:
            self.speed_cb.setCurrentIndex(list(WHISPER_SPEEDS).index(self.settings["whisper_model"]))
        self.speed_cb.setToolTip("Takes effect next time you press Start")
        self.speed_cb.currentIndexChanged.connect(lambda: self._save("whisper_model", self.speed_cb.currentData()))
        row_speed.addWidget(self.speed_cb, 1)
        row_speed.addWidget(QLabel("Run on"))
        self.device_cb = QComboBox()
        for key, label in DEVICES.items():
            self.device_cb.addItem(label, key)
        self.device_cb.setCurrentIndex(list(DEVICES).index(self.settings.get("device", "auto")))
        self.device_cb.setToolTip("Used for Whisper and the local AI model. Takes effect next time you press Start.")
        self.device_cb.currentIndexChanged.connect(lambda: self._save("device", self.device_cb.currentData()))
        row_speed.addWidget(self.device_cb)
        s.addLayout(row_speed)

        row3 = QHBoxLayout()
        self.folders_lbl = QLabel()
        row3.addWidget(self.folders_lbl, 1)
        add = QPushButton("Add folder")
        add.clicked.connect(self.add_folder)
        row3.addWidget(add)
        clear = QPushButton("Clear")
        clear.clicked.connect(self.clear_folders)
        row3.addWidget(clear)
        s.addLayout(row3)
        lay.addWidget(self.settings_box)
        self._update_folders_label()

        # answers (top) and live transcript (bottom)
        split = QSplitter(Qt.Vertical)
        self.answers = QTextBrowser()
        self.answers.setPlaceholderText("Answers appear here.")
        self.transcript_box = QPlainTextEdit(readOnly=True)
        self.transcript_box.setPlaceholderText("Live transcript appears here.")
        split.addWidget(self.answers)
        transcript_wrap = QWidget()
        tw = QVBoxLayout(transcript_wrap)
        tw.setContentsMargins(0, 0, 0, 0)
        tw.addWidget(self.transcript_box, 1)
        self.partial_lbl = QLabel()
        self.partial_lbl.setWordWrap(True)
        self.partial_lbl.setStyleSheet("color: #9aa4b2; font-style: italic; padding: 2px 6px;")
        tw.addWidget(self.partial_lbl)
        split.addWidget(transcript_wrap)
        split.setSizes([340, 160])
        lay.addWidget(split, 1)

        grip_row = QHBoxLayout()
        grip_row.addStretch(1)
        grip_row.addWidget(QSizeGrip(self))
        lay.addLayout(grip_row)

    def showEvent(self, e):
        super().showEvent(e)
        self._toggle_hidden(self.hide_chk.isChecked(), save=False)

    def _toggle_hidden(self, hidden, save=True):
        ok = set_hidden_from_capture(self, hidden)
        if hidden and not ok:
            self._set_status("Couldn't hide from capture (needs Windows 10 2004+)")
        if save:
            self._save("hide_from_share", hidden)

    def _change_source(self):
        mode = self.source_cb.currentData()
        self._save("source", mode)
        if self.transcriber and self.transcriber.is_alive():
            self.transcriber.set_mode(mode)      # live switch, no restart needed
            self._set_status(f"Now listening to: {SOURCES[mode]}")

    def _save(self, key, value):
        self.settings[key] = value
        save_settings(self.settings_path, self.settings)

    def _set_status(self, text):
        self.status_lbl.setText(text)

    # ---------------- folders ----------------
    def _update_folders_label(self):
        n = len(self.settings["folders"])
        self.folders_lbl.setText(f"{n} linked folder{'s' if n != 1 else ''}")
        self.folders_lbl.setToolTip("\n".join(self.settings["folders"]) or "None")

    def add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Link a folder (code, notes, docs)")
        if folder and folder not in self.settings["folders"]:
            self._save("folders", self.settings["folders"] + [folder])
            self._update_folders_label()
            self._reindex()

    def clear_folders(self):
        self._save("folders", [])
        self._update_folders_label()
        self._reindex()

    def _reindex(self):
        folders = list(self.settings["folders"])

        def work():
            if folders:
                self.bridge.status.emit("Indexing folders...")
            try:
                n = self.kb.build(folders)
                if folders:
                    self.bridge.status.emit(f"Indexed {n} chunks")
            except Exception as e:
                self.bridge.status.emit(f"Indexing error: {e}")

        threading.Thread(target=work, daemon=True).start()

    # ---------------- listening ----------------
    def toggle_listening(self):
        if self.transcriber and self.transcriber.is_alive():
            self.transcriber.stop()
            self.transcriber = None
            self.start_btn.setText("Start")
            return
        out_dir = self.root / "transcripts"
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"meeting_{datetime.now():%Y-%m-%d_%H-%M-%S}.txt"
        self.transcriber = LiveTranscriber(
            source_mode=self.settings["source"],
            model_size=self.settings["whisper_model"],
            device=self.settings.get("device", "auto"),
            language=self.settings["language"],
            out_path=out_path,
            on_text=self.bridge.transcript.emit,
            on_status=self.bridge.status.emit,
            on_partial=self.bridge.partial.emit,
        )
        self.transcriber.start()
        self._warm_up_provider()
        self.start_btn.setText("Stop")

    def _warm_up_provider(self):
        """Load the local model / open the cloud connection now, so the first answer isn't slow."""
        name = self.settings["provider"]
        model = self.settings.get("models", {}).get(name)

        def work():
            try:
                provider = get_provider(name, model, self.settings.get("device", "auto"))
                if hasattr(provider, "warm_up"):
                    provider.warm_up()
            except Exception as e:
                print(f"Warm-up of {name} failed: {e}")

        threading.Thread(target=work, daemon=True).start()

    def _on_partial(self, speaker, text):
        self.partials[speaker] = text
        self._render_partials()

    def _render_partials(self):
        self.partial_lbl.setText("\n".join(f"{s}: {t} …" for s, t in self.partials.items()))

    def _on_transcript(self, speaker, text):
        self.partials.pop(speaker, None)
        self._render_partials()
        self.transcript_box.appendPlainText(f"{speaker}: {text}")
        self.assistant.add_line(speaker, text)
        # With "Call + mic", only answer the other people's questions, not your own.
        from_others = self.settings["source"] != "both" or speaker == "Them"
        if self.auto_chk.isChecked() and from_others and looks_like_question(text):
            self.ask(text)

    # ---------------- answering ----------------
    def answer_latest(self):
        question = self.assistant.last_question or "Answer the most recent question in the transcript."
        self.ask(question)

    def ask(self, question):
        if self.answering:
            self.pending_question = question   # answer it as soon as the current one finishes
            return
        self.answering = True
        self._answer_finished = False
        name = self.settings["provider"]
        model = self.settings.get("models", {}).get(name)
        system, user = self.assistant.build_prompt(question)

        def work():
            self.bridge.answer_start.emit(question)
            try:
                for piece in get_provider(name, model, self.settings.get("device", "auto")).stream(system, user):
                    self.bridge.answer_chunk.emit(piece)
            except Exception as e:
                self.bridge.answer_chunk.emit(f"\n[Error from {name}: {e}]")
            self.bridge.answer_done.emit()

        threading.Thread(target=work, daemon=True).start()

    def _on_answer_start(self, question):
        self.answers.moveCursor(QTextCursor.End)
        self.answers.insertHtml(f"<p style='color:#8fb3ff'><b>Q:</b> {question}</p><br>")
        self.answers.moveCursor(QTextCursor.End)

    def _on_answer_chunk(self, piece):
        self._type_buffer += piece

    def _type_tick(self):
        if self._type_buffer:
            # reveal faster when lots of text is waiting, so typing never lags far behind
            n = max(2, len(self._type_buffer) // 12)
            out, self._type_buffer = self._type_buffer[:n], self._type_buffer[n:]
            self.answers.moveCursor(QTextCursor.End)
            self.answers.insertPlainText(out)
            self.answers.ensureCursorVisible()
        elif self._answer_finished and self.answering:
            self.answers.moveCursor(QTextCursor.End)
            self.answers.insertPlainText("\n\n")
            self.answering = False
            if self.pending_question:
                q, self.pending_question = self.pending_question, None
                self.ask(q)

    def _on_answer_done(self):
        self._answer_finished = True   # the typewriter finishes up, then frees the next question

    def closeEvent(self, e):
        if self.transcriber:
            self.transcriber.stop()
        super().closeEvent(e)
        QApplication.quit()   # we turned off auto-quit in main.py, so quit explicitly here
