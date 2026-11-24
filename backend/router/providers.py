"""
LLM router: every provider exposes the same method:

    provider.stream(system: str, user: str) -> iterator of text pieces

so the rest of the app never cares which model is answering.

FREE options (default):
  local   - Hugging Face model running on your own PC with transformers. No key, no internet after download.
  ollama  - same idea but much faster on CPU (quantized GGUF). Needs the free Ollama app installed.
  gemini  - Google AI Studio free tier (rate-limited, needs a free key; free-tier data may be used by Google).

PAID / credit-based (only used if you pick them):
  nim (free trial credits, then paid), openai, claude
"""

import os
import threading

# Display name, whether it costs money, and what it needs
PROVIDERS = {
    "local":  "Local HF model (free)",
    "ollama": "Ollama local (free, faster)",
    "gemini": "Gemini free tier (free key)",
    "nim":    "NVIDIA NIM (trial credits)",
    "openai": "OpenAI (paid)",
    "claude": "Claude (paid)",
}

DEFAULT_MODELS = {
    "local":  "Qwen/Qwen2.5-Coder-1.5B-Instruct",   # small coding model, good at code questions
    "ollama": "hf.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF:Q4_K_M",
    "gemini": "gemini-3.8-flash",   # current free-tier Flash (Nov 2026)
    "nim":    None,                 # None = ask NVIDIA which models exist and pick a good one
    "openai": "gpt-4.1-mini",
    "claude": "claude-opus-5-5",
}

MAX_ANSWER_TOKENS = 400


class LocalHFProvider:
    """Runs a Hugging Face chat model with transformers, streaming tokens as they are generated."""

    _cache = {}   # model_id -> (tokenizer, model). Load once, reuse for every question.
    _load_lock = threading.Lock()

    def __init__(self, model_id, device="auto"):
        self.device = device
        self.model_id = model_id

    def _load(self):
        with self._load_lock:
            key = (self.model_id, self.device)
            if key not in self._cache:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                use_gpu = self.device != "cpu" and torch.cuda.is_available()
                if self.device == "gpu" and not use_gpu:
                    print("GPU requested but torch can't see one (CPU-only torch?); using CPU")
                tok = AutoTokenizer.from_pretrained(self.model_id)
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_id, dtype=torch.float16 if use_gpu else torch.float32
                ).to("cuda" if use_gpu else "cpu")
                model.eval()
                self._cache[key] = (tok, model)
            return self._cache[key]

    def warm_up(self):
        self._load()

    def stream(self, system, user):
        from transformers import TextIteratorStreamer

        tok, model = self._load()
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        inputs = tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(model.device)

        # generate() blocks, so run it in a thread and read tokens from the streamer as they appear
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        worker = threading.Thread(
            target=model.generate,
            kwargs=dict(**inputs, streamer=streamer, max_new_tokens=MAX_ANSWER_TOKENS, do_sample=False),
            daemon=True,
        )
        worker.start()
        for piece in streamer:
            yield piece


# Model names change every few months (that's what caused the 404s). If the configured
# model is missing, we ask the provider's /models endpoint and take the first match below.
PREFERRED = {
    "gemini": ["gemini-3.8-flash", "gemini-3.6-flash", "flash-lite", "flash"],
    # fast-to-first-word first: in a live meeting speed beats a giant model
    "nim": ["llama-3.3-70b-instruct", "llama-3.1-8b-instruct", "gpt-oss-20b", "qwen2.5-coder",
            "qwen3-coder", "nemotron", "instruct"],
}
NOT_CHAT = ("embed", "rerank", "image", "vision", "tts", "audio", "guard", "reward", "safety",
            "live", "omni", "parse", "clip", "detector")


class OpenAICompatibleProvider:
    """OpenAI's chat API shape. Ollama, Gemini, NVIDIA NIM and OpenAI all accept it."""

    _resolved = {}   # (base_url, wanted model) -> model id that actually works
    _clients = {}    # reuse one client per server: keeps the HTTPS connection open = faster first word

    def __init__(self, model, base_url, api_key, kind=None):
        self.model, self.base_url, self.api_key, self.kind = model, base_url, api_key, kind

    def _pick_model(self, client):
        ids = [m.id.removeprefix("models/") for m in client.models.list()]
        chat_ids = [i for i in ids if not any(bad in i.lower() for bad in NOT_CHAT)]
        for want in PREFERRED.get(self.kind, []):
            for i in chat_ids:
                if want in i.lower():
                    return i
        if chat_ids:
            return chat_ids[0]
        raise RuntimeError("The provider returned no chat models for this key.")

    def _client(self):
        import openai

        ck = (self.base_url, self.api_key)
        if ck not in self._clients:
            self._clients[ck] = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._clients[ck]

    def warm_up(self):
        """Called when you press Start: opens the connection and picks the model ahead of time."""
        client = self._client()
        key = (self.base_url, self.model)
        if not (self._resolved.get(key) or self.model):
            self._resolved[key] = self._pick_model(client)
        else:
            client.models.list()   # cheap request that just opens the HTTPS connection

    def stream(self, system, user):
        import openai

        client = self._client()
        key = (self.base_url, self.model)
        model = self._resolved.get(key) or self.model
        if not model:
            model = self._resolved[key] = self._pick_model(client)
        try:
            stream = self._open(client, model, system, user)
        except openai.NotFoundError:
            old, model = model, self._pick_model(client)
            self._resolved[key] = model
            yield f"[model '{old}' not available, switched to '{model}']\n"
            stream = self._open(client, model, system, user)
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def _open(self, client, model, system, user):
        return client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=MAX_ANSWER_TOKENS,
            stream=True,
        )


class ClaudeProvider:
    def __init__(self, model, api_key):
        self.model, self.api_key = model, api_key

    def stream(self, system, user):
        import anthropic

        client = anthropic.Anthropic(api_key=self.api_key)
        with client.beta.messages.stream(
            model=self.model,
            max_tokens=4000,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": "low"},          # fast answers for a live meeting
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        ) as stream:
            for text in stream.text_stream:
                yield text


def _key(env_name):
    key = os.getenv(env_name)
    if not key:
        raise RuntimeError(f"Set {env_name} in the .env file to use this provider.")
    return key


def get_provider(name, model=None, device="auto"):
    model = model or DEFAULT_MODELS[name]
    if name == "local":
        return LocalHFProvider(model, device)
    if name == "ollama":
        return OpenAICompatibleProvider(model, "http://localhost:11434/v1", "ollama")
    if name == "gemini":
        return OpenAICompatibleProvider(
            model, "https://generativelanguage.googleapis.com/v1beta/openai/", _key("GEMINI_API_KEY"), "gemini"
        )
    if name == "nim":
        return OpenAICompatibleProvider(
            model, "https://integrate.api.nvidia.com/v1", _key("NVIDIA_API_KEY"), "nim"
        )
    if name == "openai":
        return OpenAICompatibleProvider(model, None, _key("OPENAI_API_KEY"))
    if name == "claude":
        return ClaudeProvider(model, _key("ANTHROPIC_API_KEY"))
    raise ValueError(f"Unknown provider: {name}")
