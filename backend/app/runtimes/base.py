from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class ModelRuntime(ABC):
    """Contract for a locally executing model backend."""

    def __init__(self, model_path: Path, requested_device: str = "auto") -> None:
        self.model_path = model_path
        self.requested_device = requested_device
        self.device = "cpu"
        self.model: Any = None
        self.load_seconds: float | None = None

    @abstractmethod
    def load_model(self) -> dict[str, Any]: ...

    @abstractmethod
    def unload_model(self) -> None: ...

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int, temperature: float, top_p: float) -> dict[str, Any]: ...

    def generate_stream(self, prompt: str, max_tokens: int, temperature: float, top_p: float):
        import time
        started = time.perf_counter()
        result = self.generate(prompt, max_tokens, temperature, top_p)
        yield {"type": "token", "text": result["text"]}
        elapsed = time.perf_counter() - started
        count = result.get("tokens_generated")
        yield {"type": "done", "tokens_generated": count, "generation_time": elapsed, "tokens_per_second": count / elapsed if count is not None and elapsed > 0 else None}

    @abstractmethod
    def get_metadata(self) -> dict[str, Any]: ...

    def get_memory_usage(self) -> int | None:
        return None

    def get_model_size(self) -> int:
        if self.model_path.is_file():
            return self.model_path.stat().st_size
        return sum(p.stat().st_size for p in self.model_path.rglob("*") if p.is_file())

    def get_supported_features(self) -> list[str]:
        return ["load", "metadata", "model_size"]

    def perplexity(self, text: str) -> float | None:
        return None
