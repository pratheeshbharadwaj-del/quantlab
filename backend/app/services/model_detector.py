from __future__ import annotations

import json
import hashlib
import struct
from pathlib import Path
from typing import Any


SUPPORTED = {"gguf": "GGUF / llama.cpp", "onnx": "ONNX Runtime", "tflite": "TensorFlow Lite", "lite": "TensorFlow Lite", "safetensors": "Transformers / PyTorch", "bin": "Transformers / PyTorch", "pt": "PyTorch", "pth": "PyTorch"}


def _safetensors_metadata(path: Path) -> dict[str, str]:
    try:
        with path.open("rb") as stream:
            raw_length = stream.read(8)
            if len(raw_length) != 8:
                return {}
            header_length = struct.unpack("<Q", raw_length)[0]
            if header_length > 16 * 1024 * 1024:
                return {}
            header = json.loads(stream.read(header_length).decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, struct.error):
        return {}
    metadata = header.get("__metadata__", {})
    return metadata if isinstance(metadata, dict) else {}


def detect_model(path: Path) -> dict[str, Any]:
    files = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
    if not files:
        raise ValueError("The selected model path contains no files.")
    extensions = {p.suffix.lower().lstrip(".") for p in files}
    ext = next((x for x in ("gguf", "onnx", "tflite", "lite", "safetensors", "pt", "pth", "bin") if x in extensions), None)
    config_path = path / "config.json" if path.is_dir() else None
    config: dict[str, Any] = {}
    if config_path and config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            config = {}
    safetensors_files = sorted(path.glob("*.safetensors")) if path.is_dir() else ([path] if path.suffix.lower() == ".safetensors" else [])
    checkpoint_metadata = _safetensors_metadata(safetensors_files[0]) if safetensors_files else {}
    if config and any(x in extensions for x in ("safetensors", "bin", "pt", "pth")):
        ext = "safetensors" if "safetensors" in extensions else ("bin" if "bin" in extensions else "pt")
    tokenizer_config: dict[str, Any] = {}
    tokenizer_config_path = path / "tokenizer_config.json" if path.is_dir() else None
    if tokenizer_config_path and tokenizer_config_path.is_file():
        try:
            tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            tokenizer_config = {}
    total_size = sum(p.stat().st_size for p in files)
    fingerprint = hashlib.sha256()
    model_payloads = sorted((p for p in files if p.suffix.lower() in {".gguf", ".onnx", ".tflite", ".lite", ".safetensors", ".bin", ".pt", ".pth"} or p.name.lower() == "config.json"), key=lambda p: str(p.relative_to(path.parent if path.is_file() else path)).lower())
    for model_file in model_payloads:
        fingerprint.update(model_file.suffix.lower().encode("utf-8"))
        with model_file.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                fingerprint.update(chunk)
    name = config.get("_name_or_path") or (path.stem if path.is_file() else path.name)
    architecture = ", ".join(config.get("architectures", [])) or config.get("model_type")
    params = config.get("num_parameters")
    if params is None and config.get("num_hidden_layers") and config.get("hidden_size"):
        # This is only a rough config-derived parameter estimate; avoid presenting it as exact.
        params = None
    quant = config.get("quantization_config")
    if not isinstance(quant, dict) and checkpoint_metadata.get("quantization") == "int8-per-tensor":
        quant = {"bits": 8, "quant_method": "INT8 per-tensor"}
    quant_sidecar = path / "quantization_config.json" if path.is_dir() else None
    if not isinstance(quant, dict) and quant_sidecar and quant_sidecar.is_file():
        try:
            quant = json.loads(quant_sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            quant = None
    quant_name = None
    if isinstance(quant, dict):
        bits = quant.get("bits")
        if not bits and (quant.get("load_in_4bit") or quant.get("_load_in_4bit")):
            bits = 4
        elif not bits and (quant.get("load_in_8bit") or quant.get("_load_in_8bit")):
            bits = 8
        quant_type = quant.get("bnb_4bit_quant_type")
        method = quant.get("quant_method") or quant.get("quantization_method")
        if bits and quant_type:
            quant_name = f"{bits}-bit {quant_type.upper()} ({method or 'quantized'})"
        elif bits:
            quant_name = f"{bits}-bit ({method})" if method else f"{bits}-bit"
        else:
            quant_name = str(method) if method else None
    runtime = SUPPORTED.get(ext or "")
    required_packages = _required_packages(ext, quant_name)
    missing_packages = [package for package in required_packages if not _package_installed(package)]
    installed = bool(runtime) and not missing_packages
    return {
        "format": ext.upper() if ext else "Unknown",
        "architecture": architecture,
        "name": str(name),
        "parameter_count": params,
        "quantization": quant_name,
        "size_bytes": total_size,
        "content_fingerprint": fingerprint.hexdigest() if model_payloads else None,
        "runtime": runtime or "None",
        "required_packages": required_packages,
        "missing_runtime_packages": missing_packages,
        "runtime_installed": installed,
        "supported": bool(runtime and installed),
        "generation_supported": ext == "gguf" or (ext in {"safetensors", "bin"} and bool(config)),
        "chat_template_available": bool(tokenizer_config.get("chat_template")),
        "chat_format_note": "Tokenizer declares a chat template" if tokenizer_config.get("chat_template") else ("No tokenizer chat template detected; this model may only complete text rather than follow chat instructions." if ext in {"safetensors", "bin"} else None),
        "checkpoint_metadata": checkpoint_metadata,
        "compatibility_message": "Runtime available" if runtime and installed else (f"Install required runtime packages: {', '.join(missing_packages)}" if runtime else "Unsupported model format for this device/runtime"),
    }


def _required_packages(ext: str | None, quantization: str | None) -> list[str]:
    if ext == "gguf":
        return ["llama-cpp-python"]
    if ext in {"safetensors", "bin"}:
        packages = ["torch", "transformers", "safetensors"]
        if quantization and "bitsandbytes" in quantization.lower():
            packages.extend(["bitsandbytes", "accelerate"])
        return packages
    if ext in {"pt", "pth"}:
        return ["torch"]
    if ext == "onnx":
        return ["onnxruntime"]
    if ext in {"tflite", "lite"}:
        return ["ai-edge-litert"]
    return []


def _package_installed(package: str) -> bool:
    import importlib.util
    modules = {"llama-cpp-python": "llama_cpp", "torch": "torch", "transformers": "transformers", "safetensors": "safetensors", "bitsandbytes": "bitsandbytes", "accelerate": "accelerate", "onnxruntime": "onnxruntime", "ai-edge-litert": "ai_edge_litert"}
    module = modules[package]
    if package == "ai-edge-litert" and importlib.util.find_spec(module) is None:
        return importlib.util.find_spec("tensorflow") is not None
    return importlib.util.find_spec(module) is not None
