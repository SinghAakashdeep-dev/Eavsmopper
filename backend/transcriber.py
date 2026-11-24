"""
Live transcriber: runs in a background thread.
Audio sources -> queue -> one Segmenter per speaker -> faster-whisper -> .txt file + callback.
This is your live_transcribe.py turned into a reusable class.
"""

import os
import queue
import threading
import time
from datetime import datetime

import numpy as np
from faster_whisper import WhisperModel

from .audio_sources import BLOCK_SECONDS, SAMPLE_RATE, MicSource, SystemAudioSource

MIN_SPEECH_RMS = 0.006       # never treat anything quieter than this as speech
NOISE_MULTIPLIER = 3.0       # speech = louder than 3x the background noise level
SILENCE_TO_CUT = 0.35         # pause that ends a phrase (shorter = text appears sooner)
MAX_CHUNK_SECONDS = 6.0       # long monologues get sent in 6 s pieces
MIN_SPEECH_SECONDS = 0.5
PREROLL_BLOCKS = 3
PARTIAL_EVERY_S = 0.5        # while someone is still talking, re-transcribe the phrase this often

# Whisper tends to "hear" these in noise/silence. Drop them.
HALLUCINATIONS = {"you", "thank you", "thank you.", "thanks for watching!", "bye.", "bye", "."}


def _bar(level):
    """0.0-~0.2 loudness -> a 5-step text meter like '▮▮▯▯▯'."""
    steps = min(5, int(level / 0.02 + 0.999)) if level > 0.001 else 0
    return "▮" * steps + "▯" * (5 - steps)


class Segmenter:
    """Collects blocks into phrases, cutting at pauses (same logic as live_transcribe.py)."""

    def __init__(self):
        self.preroll = []
        self._reset()
        self.last_block_time = time.monotonic()
        # Adaptive threshold: mics are often much quieter than call audio, so each source
        # learns its own background-noise level instead of sharing one fixed number.
        self.noise = 0.002
        self.phrase_id = 0   # bumps every time a phrase is finished
        self.level = 0.0     # loudest block since the UI last asked (for the level meter)

    def _reset(self):
        self.phrase, self.speech_s, self.silence_s = [], 0.0, 0.0

    def current(self):
        """The phrase heard so far (for live 'partial' text), or None."""
        if self.phrase and self.speech_s >= 0.4:
            return np.concatenate(self.phrase)
        return None

    def _finish(self):
        self.phrase_id += 1
        chunk = np.concatenate(self.phrase) if self.speech_s >= MIN_SPEECH_SECONDS else None
        self._reset()
        self.preroll = []
        return chunk

    def feed(self, block):
        self.last_block_time = time.monotonic()
        rms = float(np.sqrt(np.mean(block ** 2)))
        self.level = max(self.level, rms)
        threshold = max(MIN_SPEECH_RMS, self.noise * NOISE_MULTIPLIER)
        is_speech = rms > threshold
        if not is_speech:
            self.noise = 0.95 * self.noise + 0.05 * rms   # slowly track the quiet level
        if is_speech:
            if not self.phrase:
                self.phrase.extend(self.preroll)
            self.phrase.append(block)
            self.speech_s += BLOCK_SECONDS
            self.silence_s = 0.0
        elif self.phrase:
            self.phrase.append(block)
            self.silence_s += BLOCK_SECONDS
        else:
            self.preroll = (self.preroll + [block])[-PREROLL_BLOCKS:]
            return None

        if self.silence_s >= SILENCE_TO_CUT or len(self.phrase) * BLOCK_SECONDS >= MAX_CHUNK_SECONDS:
            return self._finish()
        return None

    def tick(self, now):
        """Loopback stops sending blocks when the call goes quiet, so also cut on wall-clock time."""
        if self.phrase and now - self.last_block_time >= SILENCE_TO_CUT:
            return self._finish()
        return None


class LiveTranscriber(threading.Thread):
    def __init__(self, source_mode, model_size, language, out_path, on_text, on_status, on_partial=None,
                 device="auto"):
        super().__init__(daemon=True)
        self.source_mode = source_mode      # "system" | "mic" | "both"
        self.model_size = model_size
        self.device = device                # "auto" | "cpu" | "gpu"
        self.language = language or None
        self.out_path = out_path
        self.on_text = on_text              # on_text(speaker, text)
        self.on_status = on_status          # on_status(message)
        self.on_partial = on_partial        # on_partial(speaker, text-so-far) while they are still talking
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def set_mode(self, mode):
        """Switch Call / Mic / Call + mic while running. Both devices stay open; we just
        ignore the one you turned off, so switching is instant."""
        self.source_mode = mode

    def _active(self, label):
        return label in {"system": {"Them"}, "mic": {"Me"}, "both": {"Me", "Them"}}[self.source_mode]

    def _transcribe(self, model, audio, fast=False):
        segments, _ = model.transcribe(
            audio,
            language=self.language,
            beam_size=1,
            condition_on_previous_text=False,
            without_timestamps=True,         # we don't need timestamps: skipping them is faster
            vad_filter=not fast,             # Silero VAD: skips non-speech inside the chunk
        )
        parts = [s.text.strip() for s in segments if s.no_speech_prob < 0.6]
        text = " ".join(parts).strip()
        return "" if text.lower() in HALLUCINATIONS else text

    def _worker(self, model, label, work_q, f, f_lock):
        """One worker per speaker, so the mic never waits behind call audio (and vice versa)."""
        while not self._stop_event.is_set():
            try:
                chunk = work_q.get(timeout=0.2)
            except queue.Empty:
                continue
            t0 = time.perf_counter()
            text = self._transcribe(model, chunk)
            took = time.perf_counter() - t0
            print(f"[{label}] {len(chunk) / SAMPLE_RATE:.1f}s audio -> {took:.2f}s  backlog {work_q.qsize()}")
            if text:
                with f_lock:
                    try:
                        f.write(f"[{datetime.now():%H:%M:%S}] {label}: {text}\n")
                        f.flush()
                    except ValueError:   # file already closed because you pressed Stop
                        pass
                self.on_text(label, text)

    def _partial_worker(self, model, label, partial_q, segmenters):
        """Gives the 'words appear while they talk' effect. Display only; the final text
        for the phrase comes from _worker once the speaker pauses."""
        while not self._stop_event.is_set():
            try:
                phrase_id, audio = partial_q.get(timeout=0.2)
            except queue.Empty:
                continue
            text = self._transcribe(model, audio, fast=True)
            # Only show it if that phrase is still in progress (the final may already be out)
            if text and self.on_partial and segmenters[label].phrase_id == phrase_id:
                self.on_partial(label, text)

    def _load_model(self):
        """GPU (CUDA) if asked/available, otherwise CPU. Falls back to CPU if the GPU fails."""
        silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
        if self.device != "cpu":
            try:
                import ctranslate2
                gpus = ctranslate2.get_cuda_device_count()
            except Exception:
                gpus = 0
            if gpus > 0:
                try:
                    self.on_status(f"Loading Whisper '{self.model_size}' on GPU...")
                    model = WhisperModel(self.model_size, device="cuda", compute_type="float16", num_workers=4)
                    self._transcribe(model, silence)   # warm-up; also proves the CUDA libraries load
                    print("Whisper is running on the GPU")
                    return model
                except Exception as e:
                    print(f"GPU failed ({e}); falling back to CPU")
                    self.on_status("GPU failed, using CPU (see terminal)")
            elif self.device == "gpu":
                print("No NVIDIA/CUDA GPU found; using CPU")
                self.on_status("No NVIDIA GPU found, using CPU")

        self.on_status(f"Loading Whisper '{self.model_size}' on CPU...")
        threads = max(2, (os.cpu_count() or 4) // 2)
        # num_workers lets the speaker threads (final + live text) transcribe at the same time
        model = WhisperModel(self.model_size, device="cpu", compute_type="int8",
                             cpu_threads=threads, num_workers=4)
        self._transcribe(model, silence)   # warm-up: the first transcription is always slow
        return model

    def run(self):
        sources = []
        try:
            model = self._load_model()

            q = queue.Queue()
            # Open BOTH devices up front so the source can be switched mid-meeting.
            for cls, label in ((MicSource, "Me"), (SystemAudioSource, "Them")):
                src = cls(label, q)
                try:
                    src.start()
                    sources.append(src)
                except Exception as e:
                    print(f"Could not open {label} audio: {e}")
            if not sources:
                raise RuntimeError("No audio device could be opened.")
            segmenters = {s.label: Segmenter() for s in sources}
            work_qs = {s.label: queue.Queue() for s in sources}
            partial_qs = {s.label: queue.Queue(maxsize=1) for s in sources}
            partial_len = {s.label: 0 for s in sources}

            with open(self.out_path, "a", encoding="utf-8") as f:
                f_lock = threading.Lock()
                for label, wq in work_qs.items():
                    threading.Thread(target=self._worker, args=(model, label, wq, f, f_lock), daemon=True).start()
                    threading.Thread(target=self._partial_worker, args=(model, label, partial_qs[label], segmenters),
                                     daemon=True).start()
                print("Audio sources ->", ", ".join(f"{s.label}: {s.device_name}" for s in sources))
                self.on_status("Listening")
                last_meter = time.monotonic()

                # This loop only cuts audio into phrases; it never waits on Whisper,
                # so pause detection stays accurate while transcription runs.
                while not self._stop_event.is_set():
                    try:
                        label, block = q.get(timeout=0.1)
                        if not self._active(label):
                            segmenters[label] = Segmenter()   # drop half-heard phrases of a muted source
                            continue
                        chunk = segmenters[label].feed(block)
                        if chunk is not None:
                            work_qs[label].put(chunk)
                    except queue.Empty:
                        pass
                    now = time.monotonic()
                    for label, seg in segmenters.items():
                        if not self._active(label):
                            continue
                        chunk = seg.tick(now)
                        if chunk is not None:
                            work_qs[label].put(chunk)
                    # Live partials: only when that speaker's final queue is idle, so they never slow finals down
                    for label, seg in segmenters.items():
                        if not self._active(label) or not seg.phrase:
                            partial_len[label] = 0
                            continue
                        n = len(seg.phrase)
                        if (n - partial_len[label]) * BLOCK_SECONDS >= PARTIAL_EVERY_S and work_qs[label].empty():
                            audio = seg.current()
                            if audio is not None and partial_qs[label].empty():
                                partial_qs[label].put((seg.phrase_id, audio))
                                partial_len[label] = n
                    if now - last_meter >= 1.0:   # level meter: proves each source is receiving sound
                        last_meter = now
                        meter = "  ".join(f"{lbl} {_bar(seg.level)}" for lbl, seg in segmenters.items()
                                          if self._active(lbl))
                        for seg in segmenters.values():
                            seg.level = 0.0
                        self.on_status(f"Listening  {meter}")
                time.sleep(0.3)   # let workers finish writing before the file closes
            self.on_status("Stopped")
        except Exception as e:  # show any error in the UI instead of dying silently
            self.on_status(f"Error: {e}")
        finally:
            for s in sources:
                try:
                    s.stop()
                except Exception:
                    pass
