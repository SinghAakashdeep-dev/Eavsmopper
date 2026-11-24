"""Small JSON settings file (settings.json next to main.py). Remembers your choices between runs."""

import json
from pathlib import Path

DEFAULTS = {
    "source": "system",          # "system" = call audio, "mic", or "both"
    "provider": "local",         # free by default
    "models": {},                # optional per-provider model override, e.g. {"local": "Qwen/..."}
    "auto_answer": True,
    "hide_from_share": True,
    "folders": [],
    "whisper_model": "base.en",  # tiny.en = fastest, base.en = fast, small.en = most accurate
    "settings_version": 2,
    "language": "en",
    "device": "auto",            # "auto" = GPU if one is found, "cpu", or "gpu"
}


def load_settings(path: Path):
    data = dict(DEFAULTS)
    if path.exists():
        try:
            data.update(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            pass
    if data.get("settings_version", 1) < 2:   # older files had the slower "small" model saved
        data["whisper_model"] = DEFAULTS["whisper_model"]
        data["settings_version"] = 2
    return data


def save_settings(path: Path, data):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
