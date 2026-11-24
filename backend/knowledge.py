"""
Linked folders -> searchable chunks (free, fully local, no embeddings model needed).

How it works:
  1. Walk every linked folder, read text/code/PDF files.
  2. Split each file into ~1200-character chunks that overlap a little.
  3. When a question comes in, score every chunk with BM25 (the classic keyword-ranking
     formula search engines used before embeddings) and return the best few.
These chunks are pasted into the LLM prompt as "reference material".
"""

import math
import os
import re
import threading
from collections import Counter
from pathlib import Path

TEXT_EXT = {
    ".txt", ".md", ".py", ".js", ".ts", ".tsx", ".jsx", ".json", ".csv", ".html", ".css",
    ".java", ".c", ".h", ".cpp", ".cs", ".go", ".rs", ".rb", ".php", ".kt", ".swift",
    ".yaml", ".yml", ".toml", ".ini", ".sql", ".sh", ".ps1",
}
SKIP_DIRS = {".git", "node_modules", "venv", ".venv", "env", "__pycache__", "dist", "build", ".next",
             "site-packages", "transcripts"}   # transcripts = this app's own meeting logs, not knowledge
SKIP_FILES = {"settings.json", "package-lock.json", "yarn.lock", ".env"}
MAX_FILE_BYTES = 20_000_000   # notebooks/Word files can be big because of embedded images
CHUNK_CHARS = 1200
OVERLAP_CHARS = 200

TOKEN_RE = re.compile(r"[a-z0-9_]+")
STOPWORDS = set("the a an and or of to in on for is are was were be it this that with as at by from what how why "
                "when where who which can could would should do does did you your we our i me my".split())


def _stem(w):
    """Tiny stemmer so 'APIs' matches 'api', 'transcribing' matches 'transcribe', etc."""
    for suffix in ("ing", "ed", "es", "s"):
        if len(w) > len(suffix) + 2 and w.endswith(suffix):
            return w[: -len(suffix)]
    return w


def tokenize(text):
    # split snake_case too: "get_provider" -> "get", "provider"
    words = TOKEN_RE.findall(text.lower().replace("_", " "))
    return [_stem(w) for w in words if len(w) > 1 and w not in STOPWORDS]


def _dedupe_folders(folders):
    """Drop folders that sit inside another linked folder (otherwise every file is indexed twice)."""
    roots = sorted({str(Path(f).resolve()) for f in folders}, key=len)
    kept = []
    for r in roots:
        if not any(Path(r).is_relative_to(k) for k in kept):
            kept.append(r)
    return kept


# Binary formats we can't read (images, media, archives, compiled files): skipped without opening
BINARY_EXT = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".mp3", ".mp4", ".wav", ".mov", ".avi",
    ".zip", ".rar", ".7z", ".gz", ".tar", ".exe", ".dll", ".so", ".pyd", ".pyc", ".bin", ".pt",
    ".safetensors", ".onnx", ".db", ".sqlite", ".woff", ".woff2", ".ttf", ".doc", ".ppt", ".xls",
}
RICH_EXT = {".pdf", ".docx", ".pptx", ".xlsx", ".ipynb"}


def _read_ipynb(path):
    """Jupyter notebook = JSON. Keep markdown and code cells, drop outputs/images."""
    import json
    nb = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    parts = []
    for cell in nb.get("cells", []):
        src = cell.get("source", "")
        src = "".join(src) if isinstance(src, list) else src
        if src.strip():
            parts.append(f"# [{cell.get('cell_type', 'cell')}]\n{src}")
    return "\n\n".join(parts)


def _read_docx(path):
    import docx                     # pip name: python-docx
    d = docx.Document(str(path))
    lines = [p.text for p in d.paragraphs]
    for table in d.tables:
        for row in table.rows:
            lines.append(" | ".join(c.text for c in row.cells))
    return "\n".join(lines)


def _read_pptx(path):
    from pptx import Presentation   # pip name: python-pptx
    lines = []
    for i, slide in enumerate(Presentation(str(path)).slides, 1):
        lines.append(f"# Slide {i}")
        for shape in slide.shapes:
            if shape.has_text_frame:
                lines.append(shape.text_frame.text)
    return "\n".join(lines)


def _read_xlsx(path):
    from openpyxl import load_workbook
    lines = []
    for ws in load_workbook(str(path), read_only=True, data_only=True).worksheets:
        lines.append(f"# Sheet {ws.title}")
        for row in ws.iter_rows(values_only=True):
            if any(v is not None for v in row):
                lines.append(" | ".join("" if v is None else str(v) for v in row))
    return "\n".join(lines)


def _looks_like_text(path):
    with open(path, "rb") as f:
        head = f.read(4096)
    return b"\x00" not in head      # binary files almost always contain NUL bytes


def read_file(path: Path):
    """Return the file's text, or None if it isn't something we can read."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader
        return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    if ext == ".ipynb":
        return _read_ipynb(path)
    if ext == ".docx":
        return _read_docx(path)
    if ext == ".pptx":
        return _read_pptx(path)
    if ext == ".xlsx":
        return _read_xlsx(path)
    if ext in BINARY_EXT:
        return None
    if path.stat().st_size > 2_000_000:   # huge logs/CSVs would swamp the search
        return None
    if ext in TEXT_EXT or _looks_like_text(path):   # any other plain-text file (.txt, .r, .vue, Dockerfile...)
        return path.read_text(encoding="utf-8", errors="ignore")
    return None


class KnowledgeBase:
    def __init__(self):
        self._lock = threading.Lock()
        self.chunks = []          # list of (source_name, text, Counter(tokens), length)
        self.df = Counter()
        self.avg_len = 1.0
        self.files = []           # relative paths of every indexed file (gives the LLM a project map)

    def build(self, folders):
        chunks, files = [], []
        for folder in _dedupe_folders(folders):
            root = Path(folder)
            if not root.is_dir():
                continue
            # os.walk skips unreadable folders (onerror) instead of crashing like rglob can
            for dirpath, dirnames, filenames in os.walk(root, onerror=lambda err: None):
                dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
                for name in filenames:
                    path = Path(dirpath) / name
                    if name in SKIP_FILES or path.suffix.lower() in BINARY_EXT:
                        continue
                    try:
                        if path.stat().st_size > MAX_FILE_BYTES:
                            continue
                        text = read_file(path)
                    except Exception:   # OneDrive placeholders, locked files, bad PDFs...
                        continue
                    if not text or not text.strip():
                        continue
                    chunks.extend(self._chunk(path, root, text))
                    files.append(str(path.relative_to(root.parent)))

        df = Counter()
        for _, _, tf, _ in chunks:
            df.update(tf.keys())
        with self._lock:
            self.chunks, self.df, self.files = chunks, df, files
            self.avg_len = (sum(c[3] for c in chunks) / len(chunks)) if chunks else 1.0
        return len(chunks)

    @staticmethod
    def _chunk(path, root, text):
        rel = str(path.relative_to(root.parent))
        out = []
        step = CHUNK_CHARS - OVERLAP_CHARS
        for start in range(0, max(len(text), 1), step):
            piece = text[start:start + CHUNK_CHARS]
            toks = tokenize(piece + " " + rel)   # file path words help matching too
            if toks:
                out.append((rel, piece, Counter(toks), len(toks)))
        return out

    def search(self, query, k=4):
        q_terms = set(tokenize(query))
        with self._lock:
            chunks, df, avg_len = self.chunks, self.df, self.avg_len
        if not q_terms or not chunks:
            return []
        n = len(chunks)
        k1, b = 1.5, 0.75
        scored = []
        for source, text, tf, length in chunks:
            score = 0.0
            for term in q_terms:
                f = tf.get(term)
                if not f:
                    continue
                idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
                score += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * length / avg_len))
            if score > 0:
                scored.append((score, source, text))
        scored.sort(reverse=True)
        return [(source, text) for _, source, text in scored[:k]]
