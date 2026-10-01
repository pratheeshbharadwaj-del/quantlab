from __future__ import annotations

import os
import platform
import sys
from typing import Any

import psutil


def system_status() -> dict[str, Any]:
    memory = psutil.virtual_memory()
    gpu: dict[str, Any] = {"available": False, "name": None, "memory_total_bytes": None, "memory_used_bytes": None}
    gpu_runtimes: list[str] = []
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            gpu = {"available": True, "name": props.name, "memory_total_bytes": props.total_memory, "memory_used_bytes": torch.cuda.memory_allocated(0)}
            gpu_runtimes.extend(["transformers", "pytorch"])
    except (ImportError, RuntimeError):
        pass
    try:
        from llama_cpp import llama_supports_gpu_offload
        if llama_supports_gpu_offload():
            gpu_runtimes.append("gguf")
            if not gpu["available"]:
                gpu["available"] = True
                gpu["name"] = "GPU acceleration available through llama.cpp"
    except (ImportError, RuntimeError):
        pass
    try:
        import onnxruntime as ort
        if "CUDAExecutionProvider" in ort.get_available_providers():
            gpu_runtimes.append("onnx")
            if not gpu["available"]:
                gpu["available"] = True
                gpu["name"] = "CUDA provider available through ONNX Runtime"
    except (ImportError, RuntimeError):
        pass
    gpu["supported_runtimes"] = gpu_runtimes
    return {"cpu_percent": psutil.cpu_percent(interval=0.1), "cpu_cores": os.cpu_count(), "ram_total_bytes": memory.total, "ram_available_bytes": memory.available, "ram_used_bytes": memory.used, "gpu": gpu, "operating_system": platform.platform(), "python_version": sys.version.split()[0], "backend_runtime": "FastAPI / Python", "process_rss_bytes": psutil.Process().memory_info().rss}
