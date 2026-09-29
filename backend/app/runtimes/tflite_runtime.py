from __future__ import annotations

import time
from typing import Any

from .base import ModelRuntime


class TFLiteRuntime(ModelRuntime):
    def load_model(self) -> dict[str, Any]:
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            try:
                from tensorflow.lite import Interpreter
            except ImportError as exc:
                raise RuntimeError("TensorFlow Lite runtime dependency is missing. Install ai-edge-litert or tensorflow.") from exc
        if not self.model_path.is_file() or self.model_path.suffix.lower() not in {".tflite", ".lite"}:
            raise ValueError("Select a .tflite model file.")
        started = time.perf_counter()
        try:
            self.model = Interpreter(model_path=str(self.model_path))
            self.model.allocate_tensors()
        except Exception as exc:
            raise RuntimeError(f"TensorFlow Lite model could not be loaded: {exc}") from exc
        self.load_seconds = time.perf_counter() - started
        return self.get_metadata()

    def unload_model(self) -> None:
        self.model = None

    def generate(self, prompt: str, max_tokens: int, temperature: float, top_p: float) -> dict[str, Any]:
        raise RuntimeError("TensorFlow Lite models do not provide a universal text-generation interface. This model can be inspected but text generation is unsupported by this adapter.")

    def get_metadata(self) -> dict[str, Any]:
        if self.model is None:
            return {}
        def tensor_details(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return [{"name": item.get("name"), "shape": item.get("shape", []).tolist(), "dtype": str(item.get("dtype")), "index": int(item.get("index", -1))} for item in items]
        return {"inputs": tensor_details(self.model.get_input_details()), "outputs": tensor_details(self.model.get_output_details())}

    def get_supported_features(self) -> list[str]:
        return ["load", "metadata", "model_size"]
