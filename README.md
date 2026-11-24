# Eavsmopper

A small always-on-top desktop overlay for Windows that **listens to your meetings, transcribes them live, and answers questions using your own files and code**.

It runs speech-to-text on your own laptop with [Whisper](https://github.com/openai/whisper) (through `faster-whisper`), and answers with either a **free local Hugging Face model** or a cloud model of your choice (Gemini free tier, NVIDIA NIM, OpenAI, Claude).

## What you can use it for

- **Project demos, reviews and viva/interviews about your own work.** Link your project folder and get bullet-point answers to questions like "which APIs did you use?" or "how does the transcriber work?", grounded in your actual code and docs.
- **Stand-ups and team calls.** Get a live transcript of everything said, saved to a `.txt` file per meeting.
- **Meeting notes.** Every meeting is saved as `transcripts/meeting_<date>_<time>.txt`, with timestamps and who spoke ("Me" vs "Them").
- **Accessibility.** Live captions for any call or video playing on your PC.

> **Please be fair and legal.** Tell people in the meeting that you are transcribing; many places require consent to record. Don't use it where outside help is not allowed (exams, assessments, interviews that forbid it).

## Features

- **Listens to call audio, your mic, or both.** "Call audio" records whatever your PC is playing (Teams, Google Meet, Zoom, YouTube...) via Windows WASAPI loopback. You can switch sources mid-meeting.
- **Live transcription.** Grey "live" words appear while people are still talking, then get replaced by the final line when they pause.
- **Answers from your files.** Link folders with code, notes, PDFs, Word (`.docx`), PowerPoint (`.pptx`), Excel (`.xlsx`), Jupyter notebooks (`.ipynb`) or any plain-text file. Questions are matched against them with BM25 keyword search.
- **Auto-answer.** When it hears a question (from "Them" in Call + mic mode), it streams an answer into the overlay, typewriter style. The **Answer** button forces an answer to the latest question.
- **Hidden from screen share.** On Windows 10 (version 2004) or newer, the overlay is excluded from screen sharing and recordings, so only you see it. This can be turned off.
- **Free by default.** The default AI runs locally with no API key. Paid providers are only used if you pick them.
- **CPU or GPU.** Runs on any laptop CPU. Uses an NVIDIA GPU automatically if one is set up.

## Requirements

- **Windows 10 (2004+) or Windows 11.** Call-audio capture and screen-share hiding are Windows-only.
- **Python 3.10 or newer (64-bit).** Get it from [python.org](https://www.python.org/downloads/) and tick "Add Python to PATH" during install.
- About **4 GB of free disk** for the models (downloaded automatically on first use).
- Optional: an NVIDIA GPU for faster transcription and local answers.

## Setup on your laptop

Open **PowerShell** and run:

```powershell
# 1. Get the code
git clone https://github.com/SinghAakashdeep-dev/Eavsmopper.git
cd Eavsmopper

# 2. Create and activate a virtual environment (inside backend/)
python -m venv backend\venv
backend\venv\Scripts\Activate.ps1

# 3. Install dependencies
python -m pip install -r backend\requirements.txt

# 4. Run it
python main.py
```

If PowerShell says *"running scripts is disabled"* at step 2, run this once and try again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

> **Tip:** always use `python -m pip ...` instead of `pip ...`. Some Windows security settings (Smart App Control) block the `pip.exe` launcher inside a venv.

### First run

1. A small dark window appears, always on top. Drag it by its title bar and resize it from the bottom-right corner.
2. Click **⚙** to open settings:
   - **Listen to:** Call audio / Microphone / Call + mic
   - **AI:** which model answers (see below)
   - **Transcription:** Fastest (`tiny.en`), Fast (`base.en`, default) or Accurate (`small.en`)
   - **Run on:** Auto / CPU / GPU
   - **Add folder:** link the folders the AI should answer from
3. Press **Start**, join your call, and watch the transcript. The status bar shows a level meter for each source (`Me ▮▮▯▯▯  Them ▮▮▮▯▯`).

The first start downloads the Whisper model (about 150 MB), and the first local answer downloads the local AI model (about 3 GB).

## Choosing the AI

| Option in ⚙ | Cost | Needs | Notes |
|---|---|---|---|
| Local HF model (default) | Free | Nothing | `Qwen/Qwen2.5-Coder-1.5B-Instruct` runs on your PC. Private, but slow on CPU (about 10-30 s per answer). |
| Ollama local | Free | [Ollama](https://ollama.com) installed | Same Qwen model, quantized: much faster on CPU. |
| Gemini free tier | Free (rate-limited) | `GEMINI_API_KEY` | Get a key at [Google AI Studio](https://aistudio.google.com/apikey). Free-tier data may be used by Google. |
| NVIDIA NIM | Trial credits | `NVIDIA_API_KEY` | Get a key at [build.nvidia.com](https://build.nvidia.com). The app picks an available model automatically. |
| OpenAI | Paid | `OPENAI_API_KEY` | |
| Claude | Paid | `ANTHROPIC_API_KEY` | |

To use a cloud option, copy `.env.example` to `.env` (in the project root) and fill in only the keys you need. **Never commit `.env`;** it is already in `.gitignore`.

If a provider retires a model, the app asks the provider for its current model list and switches automatically. You can also pin a model in `settings.json`, for example `"models": {"gemini": "gemini-3.8-flash"}`.

## Using a GPU (optional)

Set **Run on** to *Auto* or *GPU*. You also need:

1. The CUDA build of PyTorch (for the local AI): pick your command at [pytorch.org](https://pytorch.org/get-started/locally/).
2. The CUDA libraries for Whisper:
   ```powershell
   python -m pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
   ```

If the GPU can't be used, the app falls back to CPU and prints the reason in the terminal.

## Troubleshooting

| Problem | Fix |
|---|---|
| `pip.exe ... blocked by Application Control` | Use `python -m pip install ...` |
| **Me** level meter stays empty | Set your mic as the default input in Windows Settings > Sound > Input, then press Start again. The terminal prints which devices were opened (`Audio sources -> ...`). |
| **Them** stays empty | Call audio is taken from your *default output* device. Make sure the call plays through it. |
| Noise is transcribed as words | Move away from the noise, or pick a less sensitive mic. Whisper sometimes "hears" words in silence; common ones are filtered. |
| Transcription feels slow | Choose **Transcription: Fastest**. Each phrase prints its timing in the terminal (`[Them] 2.1s audio -> 0.40s`). |
| Answers ignore my files | Check the folder count in ⚙. Only text, code, PDF, docx, pptx, xlsx and ipynb files are read. `venv`, `node_modules`, `.git` and the app's own `transcripts` folder are skipped. |
| Overlay still shows in screen share | Needs Windows 10 version 2004 or newer, and "Hide from screen share" ticked. |

## How it works

```
 mic ──┐                                        ┌─> transcripts/meeting_*.txt
       ├─> 100 ms blocks ─> pause detection ─> Whisper ─┤
 call ─┘   (16 kHz mono)    (per speaker)               └─> overlay transcript
                                                                  │ question?
 linked folders ─> chunks ─> BM25 search ─> prompt ─> LLM (router) ─> streamed answer
```

```
main.py                     starts the app
frontend/overlay.py         the PySide6 overlay window (UI, settings, typewriter answers)
backend/audio_sources.py    mic + WASAPI loopback capture (PyAudioWPatch), resampled to 16 kHz
backend/transcriber.py      pause-based phrase cutting, live partials, faster-whisper workers
backend/knowledge.py        reads linked folders and does BM25 search
backend/assistant.py        question detection and prompt building
backend/router/providers.py one stream() interface for local HF, Ollama, Gemini, NIM, OpenAI, Claude
backend/settings.py         settings.json load/save
```

The early learning scripts are kept too: `backend/rec_audio.py` (record the mic to a wav file), `backend/whisper_connect.py` (transcribe a wav file with transformers) and `backend/live_transcribe.py` (first live mic transcriber).

## Roadmap

- Open automatically when a Google Meet or Teams call starts
- Semantic (embedding) search over linked folders
- Speaker names instead of Me / Them
