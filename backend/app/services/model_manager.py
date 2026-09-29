from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import psutil

from app.runtimes.gguf_runtime import GGUFRuntime
from app.runtimes.onnx_runtime import ONNXRuntime
from app.runtimes.pytorch_runtime import PyTorchRuntime
from app.runtimes.tflite_runtime import TFLiteRuntime
from app.runtimes.transformers_runtime import TransformersRuntime
from app.services.archive_import import unpack_model_archive
from app.services.model_detector import detect_model


class ModelManager:
    def __init__(self, model_root: Path) -> None:
        self.model_root = model_root.resolve()
        self.records: dict[str, dict[str, Any]] = {}
        self.loaded: dict[str, Any] = {}

    def resolve_path(self, relative_path: str) -> Path:
        candidate = (self.model_root / relative_path).resolve()
        if candidate != self.model_root and self.model_root not in candidate.parents:
            raise ValueError("Model paths must remain inside the configured local models directory.")
        if not candidate.exists():
            raise ValueError(f"Model path not found under the model directory: {relative_path}")
        return candidate

    def import_model(self, relative_path: str) -> dict[str, Any]:
        path = self.resolve_path(relative_path)
        archive_source = None
        if path.is_file() and path.suffix.lower() == ".zip":
            archive_source = str(path.relative_to(self.model_root))
            path = unpack_model_archive(path, self.model_root)
        info = detect_model(path)
        model_id = str(uuid.uuid4())
        record = {"id": model_id, "path": str(path.relative_to(self.model_root)), **info}
        if archive_source:
            record["archive_source"] = archive_source
        self.records[model_id] = record
        return record

    def get(self, model_id: str) -> dict[str, Any]:
        try:
            return self.records[model_id]
        except KeyError as exc:
            raise KeyError(f"Unknown model id: {model_id}") from exc

    def load(self, model_id: str) -> dict[str, Any]:
        record = self.get(model_id)
        if model_id in self.loaded:
            return {**record, "loaded": True, "runtime_metadata": self.loaded[model_id].get_metadata(), "load_seconds": self.loaded[model_id].load_seconds}
        path = self.resolve_path(record["path"])
        classes = {"GGUF": GGUFRuntime, "ONNX": ONNXRuntime, "TFLITE": TFLiteRuntime, "LITE": TFLiteRuntime, "SAFETENSORS": TransformersRuntime, "BIN": TransformersRuntime, "PT": PyTorchRuntime, "PTH": PyTorchRuntime}
        runtime_cls = classes.get(record["format"])
        if runtime_cls is None:
            raise RuntimeError("Unsupported model format for this device/runtime.")
        runtime = runtime_cls(path)
        runtime.ram_before_load_bytes = psutil.Process().memory_info().rss
        metadata = runtime.load_model()
        runtime.ram_after_load_bytes = psutil.Process().memory_info().rss
        runtime.ram_increase_bytes = max(0, runtime.ram_after_load_bytes - runtime.ram_before_load_bytes)
        self.loaded[model_id] = runtime
        record["loaded"] = True
        record["runtime_metadata"] = metadata
        record["load_seconds"] = runtime.load_seconds
        record["ram_before_load_bytes"] = runtime.ram_before_load_bytes
        record["ram_after_load_bytes"] = runtime.ram_after_load_bytes
        record["ram_increase_bytes"] = runtime.ram_increase_bytes
        for field in ("name", "architecture", "quantization", "parameter_count", "chat_template_available"):
            if metadata.get(field) is not None:
                record[field] = metadata[field]
        return record

    def unload(self, model_id: str | None = None) -> dict[str, Any]:
        ids = [model_id] if model_id else list(self.loaded)
        unloaded = []
        for current_id in ids:
            runtime = self.loaded.pop(current_id, None)
            if runtime is not None:
                runtime.unload_model()
                self.records[current_id]["loaded"] = False
                unloaded.append(current_id)
        return {"unloaded": unloaded}

    def generate(self, model_id: str, prompt: str, max_tokens: int, temperature: float, top_p: float) -> tuple[dict[str, Any], Any]:
        runtime = self.loaded.get(model_id)
        if runtime is None:
            raise RuntimeError("Load this model before generating text.")
        import time
        started = time.perf_counter()
        output = runtime.generate(prompt, max_tokens, temperature, top_p)
        elapsed = time.perf_counter() - started
        tokens = output.get("tokens_generated")
        return {**output, "generation_time": elapsed, "tokens_per_second": tokens / elapsed if tokens is not None and elapsed > 0 else None}, runtime

    def list_models(self) -> list[dict[str, Any]]:
        return [{**record, "loaded": model_id in self.loaded} for model_id, record in self.records.items()]

    def list_files(self, relative_path: str = "") -> list[dict[str, Any]]:
        directory = self.model_root if not relative_path else self.resolve_path(relative_path)
        if not directory.is_dir():
            raise ValueError("The selected browse path is not a directory.")
        allowed_files = {".zip", ".gguf", ".onnx", ".tflite", ".lite", ".safetensors", ".bin", ".pt", ".pth"}
        results = []
        for entry in sorted(directory.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
            if entry.name.startswith(".") or entry.is_symlink():
                continue
            if not entry.is_dir() and entry.suffix.lower() not in allowed_files:
                continue
            results.append({"name": entry.name, "path": str(entry.relative_to(self.model_root)), "is_directory": entry.is_dir(), "size_bytes": entry.stat().st_size if entry.is_file() else None})
        return results
