from __future__ import annotations

import time
from typing import Any

from .base import ModelRuntime


class PyTorchRuntime(ModelRuntime):
    """Safely inspect standalone PyTorch tensor checkpoints; no pickle execution."""

    def load_model(self) -> dict[str, Any]:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch runtime dependency is missing. Install torch.") from exc
        if not self.model_path.is_file() or self.model_path.suffix.lower() not in {".pt", ".pth"}:
            raise ValueError("Select a .pt or .pth checkpoint.")
        if self.requested_device == "gpu" and not torch.cuda.is_available():
            raise RuntimeError("GPU was selected, but this Python environment has no CUDA-enabled PyTorch runtime. Install a compatible CUDA build of PyTorch and reload the backend.")
        self.device = "cuda:0" if self.requested_device == "gpu" or (self.requested_device == "auto" and torch.cuda.is_available()) else "cpu"
        started = time.perf_counter()
        try:
            self.model = torch.load(self.model_path, map_location=self.device, weights_only=True)
        except Exception as exc:
            raise RuntimeError(f"Checkpoint could not be safely inspected as a weights-only file: {exc}") from exc
        self.load_seconds = time.perf_counter() - started
        return self.get_metadata()

    def unload_model(self) -> None:
        self.model = None

    def generate(self, prompt: str, max_tokens: int, temperature: float, top_p: float) -> dict[str, Any]:
        raise RuntimeError("A PyTorch state-dict checkpoint does not include enough architecture/tokenizer information for generic generation. Import a complete local Transformers model directory or GGUF model.")

    def get_metadata(self) -> dict[str, Any]:
        return {"checkpoint_type": type(self.model).__name__ if self.model is not None else None, "architecture": None, "device": self.device, "note": "Safely inspected weights-only checkpoint; generation requires a model architecture and tokenizer."}

    def get_supported_features(self) -> list[str]:
        return ["load", "metadata", "model_size"]
