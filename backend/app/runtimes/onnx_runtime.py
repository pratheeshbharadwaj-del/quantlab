from __future__ import annotations

import time
from typing import Any

from .base import ModelRuntime


class ONNXRuntime(ModelRuntime):
    def load_model(self) -> dict[str, Any]:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("ONNX Runtime dependency is missing. Install onnxruntime.") from exc
        if not self.model_path.is_file() or self.model_path.suffix.lower() != ".onnx":
            raise ValueError("Select a .onnx model file.")
        started = time.perf_counter()
        try:
            self.model = ort.InferenceSession(str(self.model_path), providers=ort.get_available_providers())
        except Exception as exc:
            raise RuntimeError(f"ONNX model could not be loaded: {exc}") from exc
        self.load_seconds = time.perf_counter() - started
        return self.get_metadata()

    def unload_model(self) -> None:
        self.model = None

    def generate(self, prompt: str, max_tokens: int, temperature: float, top_p: float) -> dict[str, Any]:
        raise RuntimeError("Generic ONNX graphs do not define a universal text-generation interface. This model can be inspected but text generation is unsupported by this adapter.")

    def get_metadata(self) -> dict[str, Any]:
        if self.model is None:
            return {}
        return {"inputs": [{"name": x.name, "shape": x.shape, "type": x.type} for x in self.model.get_inputs()], "outputs": [{"name": x.name, "shape": x.shape, "type": x.type} for x in self.model.get_outputs()], "providers": self.model.get_providers(), "custom_metadata": self.model.get_modelmeta().custom_metadata_map}

    def get_supported_features(self) -> list[str]:
        return ["load", "metadata", "model_size"]
