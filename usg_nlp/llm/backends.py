"""LLM backends: one interface, several ways to run an open-weight model.

All backends take chat ``messages`` plus a JSON Schema and return the raw text
the model produced.  Where the server supports it, the schema is enforced at
decode time (grammar-constrained decoding), which removes most malformed
outputs before the verifier ever sees them.

=================  ===========================================================
``ollama``         Laptop / workstation. ``ollama pull qwen2.5:7b-instruct``.
                   Uses ``/api/chat`` with ``format=<json schema>``.
``openai``         Any OpenAI-compatible server: vLLM on the Ada cluster,
                   llama.cpp ``llama-server``, LM Studio.  Sends
                   ``response_format={"type": "json_schema", ...}`` and degrades
                   to ``json_object`` / plain if the server rejects it.
``transformers``   In-process Hugging Face model (GPU recommended). Optional
                   dependency: ``pip install transformers torch accelerate``.
``replay``         Answers only from the response cache - reproduces a past run
                   exactly, with no model available (grading, CI, the dashboard).
``mock``           Scripted responses for unit tests.
=================  ===========================================================

Only the standard library is used for HTTP, so the core package gains no
dependencies.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional

LOGGER = logging.getLogger(__name__)

Message = dict[str, str]


class BackendError(RuntimeError):
    """The model server could not be reached or returned an unusable response."""


class LLMBackend(ABC):
    name: str = "base"

    def __init__(self, model: str, temperature: float = 0.0, seed: int = 7, max_tokens: int = 4096,
                 timeout: float = 300.0) -> None:
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.max_tokens = max_tokens
        self.timeout = timeout

    @abstractmethod
    def complete(self, messages: list[Message], schema: Optional[dict] = None) -> str:
        """Return the model's raw text for a chat conversation."""

    @property
    def identity(self) -> str:
        return f"{self.name}:{self.model}"

    def available(self) -> tuple[bool, str]:
        """Cheap reachability probe used by the CLI and dashboard."""
        return True, "ok"

    # ------------------------------------------------------------------ helpers
    def _post_json(self, url: str, payload: dict, headers: Optional[dict] = None) -> dict:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=body, method="POST",
                                     headers={"Content-Type": "application/json", **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise BackendError(f"HTTP {exc.code} from {url}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            raise BackendError(f"Cannot reach {url}: {exc}") from exc

    def _get(self, url: str) -> bool:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                return resp.status < 500
        except Exception:  # noqa: BLE001 - probe only
            return False


# --------------------------------------------------------------------------- Ollama


class OllamaBackend(LLMBackend):
    name = "ollama"

    def __init__(self, model: str, base_url: str = "http://localhost:11434", num_ctx: int = 16384, **kw) -> None:
        super().__init__(model, **kw)
        self.base_url = base_url.rstrip("/")
        self.num_ctx = num_ctx

    def available(self) -> tuple[bool, str]:
        ok = self._get(f"{self.base_url}/api/tags")
        return ok, "ok" if ok else f"Ollama not reachable at {self.base_url} (run `ollama serve`)"

    def complete(self, messages: list[Message], schema: Optional[dict] = None) -> str:
        payload = {
            "model": self.model, "messages": messages, "stream": False,
            "options": {"temperature": self.temperature, "seed": self.seed, "num_ctx": self.num_ctx,
                        "num_predict": self.max_tokens},
        }
        if schema is not None:
            payload["format"] = schema  # Ollama >= 0.5 enforces a JSON schema natively
        data = self._post_json(f"{self.base_url}/api/chat", payload)
        try:
            return data["message"]["content"]
        except KeyError as exc:
            raise BackendError(f"Unexpected Ollama response: {str(data)[:300]}") from exc


# --------------------------------------------------------------------------- OpenAI-compatible (vLLM etc.)


class OpenAICompatibleBackend(LLMBackend):
    name = "openai"
    _MODES = ("json_schema", "json_object", "none")

    def __init__(self, model: str, base_url: str = "http://localhost:8000/v1", api_key: Optional[str] = None, **kw) -> None:
        super().__init__(model, **kw)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._mode_idx = 0
        self._lock = threading.Lock()

    def available(self) -> tuple[bool, str]:
        ok = self._get(f"{self.base_url}/models")
        return ok, "ok" if ok else f"No OpenAI-compatible server at {self.base_url}"

    def complete(self, messages: list[Message], schema: Optional[dict] = None) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        while True:
            mode = self._MODES[self._mode_idx] if schema is not None else "none"
            payload = {"model": self.model, "messages": messages, "temperature": self.temperature,
                       "seed": self.seed, "max_tokens": self.max_tokens}
            if mode == "json_schema":
                payload["response_format"] = {"type": "json_schema",
                                              "json_schema": {"name": "usg_findings", "schema": schema, "strict": True}}
            elif mode == "json_object":
                payload["response_format"] = {"type": "json_object"}
            try:
                data = self._post_json(f"{self.base_url}/chat/completions", payload, headers)
                return data["choices"][0]["message"]["content"]
            except BackendError as exc:
                # Older servers reject json_schema: degrade once per mode, remember the choice.
                if "HTTP 400" in str(exc) and mode != "none":
                    with self._lock:
                        self._mode_idx = min(self._mode_idx + 1, len(self._MODES) - 1)
                    LOGGER.warning("Server rejected response_format=%s; falling back to %s", mode,
                                   self._MODES[self._mode_idx])
                    continue
                raise
            except (KeyError, IndexError) as exc:
                raise BackendError(f"Unexpected response: {str(data)[:300]}") from exc


# --------------------------------------------------------------------------- Hugging Face transformers


class TransformersBackend(LLMBackend):
    name = "transformers"

    def __init__(self, model: str, device_map: str = "auto", **kw) -> None:
        super().__init__(model, **kw)
        self.device_map = device_map
        self._pipe = None

    def available(self) -> tuple[bool, str]:
        try:
            import transformers  # noqa: F401
            import torch  # noqa: F401
        except ImportError:
            return False, "pip install transformers torch accelerate"
        return True, "ok"

    def _load(self):
        if self._pipe is None:
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:
                raise BackendError("transformers backend needs: pip install transformers torch accelerate") from exc
            tok = AutoTokenizer.from_pretrained(self.model)
            mdl = AutoModelForCausalLM.from_pretrained(
                self.model, device_map=self.device_map,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
            self._pipe = (tok, mdl, torch)
        return self._pipe

    def complete(self, messages: list[Message], schema: Optional[dict] = None) -> str:
        tok, mdl, torch = self._load()
        torch.manual_seed(self.seed)
        inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(mdl.device)
        with torch.no_grad():
            out = mdl.generate(inputs, max_new_tokens=self.max_tokens, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True)


# --------------------------------------------------------------------------- cache, replay, mock


class ResponseCache:
    """Append-only JSONL cache keyed by (model identity, messages, schema)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._store: dict[str, str] = {}
        self._lock = threading.Lock()
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self._store[rec["key"]] = rec["response"]

    @staticmethod
    def key(identity: str, messages: list[Message], schema: Optional[dict]) -> str:
        blob = json.dumps({"id": identity, "m": messages, "s": schema}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[str]:
        return self._store.get(key)

    def put(self, key: str, response: str, meta: Optional[dict] = None) -> None:
        with self._lock:
            self._store[key] = response
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"key": key, "response": response, "meta": meta or {},
                                     "ts": time.time()}, ensure_ascii=False) + "\n")

    def __len__(self) -> int:
        return len(self._store)


class ReplayBackend(LLMBackend):
    """Serves responses from a cache recorded with the same model identity; never calls a model."""

    name = "replay"

    def __init__(self, model: str, cache: ResponseCache, replay_identity: str, **kw) -> None:
        super().__init__(model, **kw)
        self.cache = cache
        self.replay_identity = replay_identity

    @property
    def identity(self) -> str:
        return self.replay_identity

    def complete(self, messages: list[Message], schema: Optional[dict] = None) -> str:
        hit = self.cache.get(ResponseCache.key(self.replay_identity, messages, schema))
        if hit is None:
            raise BackendError("Replay cache miss - this prompt was not recorded for this model")
        return hit


class MockBackend(LLMBackend):
    """Test double: ``responder(messages) -> str`` scripts the model's answer."""

    name = "mock"

    def __init__(self, responder: Callable[[list[Message]], str], model: str = "mock-model", **kw) -> None:
        super().__init__(model, **kw)
        self.responder = responder
        self.calls: list[list[Message]] = []

    def complete(self, messages: list[Message], schema: Optional[dict] = None) -> str:
        self.calls.append(messages)
        return self.responder(messages)


def make_backend(kind: str, model: str, base_url: Optional[str] = None, api_key: Optional[str] = None,
                 cache: Optional[ResponseCache] = None, replay_identity: Optional[str] = None, **kw) -> LLMBackend:
    kind = kind.lower()
    if kind == "ollama":
        return OllamaBackend(model, base_url=base_url or "http://localhost:11434", **kw)
    if kind in {"openai", "vllm", "llamacpp", "lmstudio"}:
        return OpenAICompatibleBackend(model, base_url=base_url or "http://localhost:8000/v1", api_key=api_key, **kw)
    if kind in {"transformers", "hf"}:
        return TransformersBackend(model, **kw)
    if kind == "replay":
        if cache is None or replay_identity is None:
            raise ValueError("replay backend needs a cache file and the identity of the recorded model")
        return ReplayBackend(model, cache, replay_identity, **kw)
    raise ValueError(f"Unknown backend {kind!r}")
