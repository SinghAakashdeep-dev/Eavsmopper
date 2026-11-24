"""
Live transcription: microphone -> Whisper -> .txt file, in near real time.

Run (venv active, inside backend/):
    python live_transcribe.py
Press Ctrl+C to stop. Transcripts go to backend/transcripts/meeting_<date>_<time>.txt

How it works (read this before the code):
  1. The mic delivers audio in small 100 ms blocks via a callback (same idea as rec_audio.py).
  2. We measure how loud each block is (RMS). Loud = speech, quiet = silence.
  3. We collect blocks into a "phrase" while you talk. When you pause (~0.6 s of silence),
     or the phrase gets too long (8 s), we send that phrase to Whisper.
     Cutting at pauses means we rarely chop a word in half.
  4. The text is appended to the .txt file immediately, with a timestamp,
     and we print how long Whisper took so you can see if it keeps up.
"""

import queue
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

# ---------------- settings you can tune ----------------
MODEL_ID = "openai/whisper-small"   # CPU friendly. Try "openai/whisper-base" for more speed.
LANGUAGE = "en"                     # set to None if you use an English-only ".en" model
SAMPLE_RATE = 16000                 # Whisper expects 16 kHz
BLOCK_SECONDS = 0.1                 # size of each mic block
SILENCE_RMS = 0.01                  # loudness threshold; raise it if background noise counts as speech
SILENCE_TO_CUT = 0.6                # pause length (s) that ends a phrase
MAX_CHUNK_SECONDS = 8.0             # force a cut if you talk this long without pausing
MIN_SPEECH_SECONDS = 0.5            # ignore clicks/coughs shorter than this
PREROLL_BLOCKS = 3                  # keep 0.3 s before speech starts so the first word isn't clipped
# --------------------------------------------------------


def load_model():
    """Load Whisper ONCE. This is the slow part; the live loop reuses it."""
    use_gpu = torch.cuda.is_available()
    device = "cuda" if use_gpu else "cpu"
    dtype = torch.float16 if use_gpu else torch.float32

    print(f"Loading {MODEL_ID} on {device} ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_ID, dtype=dtype).to(device)
    model.eval()  # inference mode (turns off training-only behaviour)
    return processor, model


def transcribe(processor, model, audio: np.ndarray) -> str:
    """Same steps as whisper_connect.py, wrapped in a function."""
    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    features = inputs.input_features.to(model.device, dtype=model.dtype)

    gen_kwargs = {"task": "transcribe"}
    if LANGUAGE:
        gen_kwargs["language"] = LANGUAGE  # skips language detection = faster

    with torch.no_grad():
        ids = model.generate(features, **gen_kwargs)
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


def main():
    processor, model = load_model()

    # Warm-up: the very first generate() is always slower, so do it before you start talking.
    transcribe(processor, model, np.zeros(SAMPLE_RATE, dtype=np.float32))

    out_dir = Path("transcripts")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"meeting_{datetime.now():%Y-%m-%d_%H-%M-%S}.txt"

    q = queue.Queue()

    def callback(indata, frames, time_info, status):
        """Runs on the audio thread. Keep it tiny: just hand the block to the main thread."""
        if status:
            print(status, flush=True)
        q.put(indata[:, 0].copy())  # take channel 0 -> 1-D array

    phrase = []                              # blocks of the phrase being collected
    preroll = deque(maxlen=PREROLL_BLOCKS)   # last few quiet blocks before speech
    speech_s = 0.0
    silence_s = 0.0

    with open(out_path, "a", encoding="utf-8") as f, sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=int(SAMPLE_RATE * BLOCK_SECONDS),
        callback=callback,
    ):
        print("#" * 70)
        print(f"Listening... writing to {out_path}")
        print("Press Ctrl+C to stop")
        print("#" * 70)

        while True:
            block = q.get()
            rms = float(np.sqrt(np.mean(block ** 2)))
            is_speech = rms > SILENCE_RMS

            if is_speech:
                if not phrase:
                    phrase.extend(preroll)  # include the moment just before you started talking
                phrase.append(block)
                speech_s += BLOCK_SECONDS
                silence_s = 0.0
            elif phrase:
                phrase.append(block)        # silence inside/after a phrase
                silence_s += BLOCK_SECONDS
            else:
                preroll.append(block)       # silence with no phrase going: just remember it
                continue

            phrase_len = len(phrase) * BLOCK_SECONDS
            if silence_s < SILENCE_TO_CUT and phrase_len < MAX_CHUNK_SECONDS:
                continue  # still talking, keep collecting

            # ---- phrase finished: transcribe it ----
            if speech_s >= MIN_SPEECH_SECONDS:
                audio = np.concatenate(phrase)
                t0 = time.perf_counter()
                text = transcribe(processor, model, audio)
                took = time.perf_counter() - t0
                audio_s = len(audio) / SAMPLE_RATE

                if text:
                    stamp = datetime.now().strftime("%H:%M:%S")
                    f.write(f"[{stamp}] {text}\n")
                    f.flush()  # make it appear in the file right now, not at exit
                    print(f"[{stamp}] {text}")

                # RTF < 1.0 means faster than real time. backlog = audio waiting in the queue.
                backlog_s = q.qsize() * BLOCK_SECONDS
                print(f"    {audio_s:.1f}s audio -> {took:.2f}s  (RTF {took / audio_s:.2f}, backlog {backlog_s:.1f}s)")

            phrase = []
            preroll.clear()
            speech_s = 0.0
            silence_s = 0.0


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped. Transcript saved in the transcripts/ folder.")
