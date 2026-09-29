from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .base import ModelRuntime


class GGUFRuntime(ModelRuntime):
    def load_model(self) -> dict[str, Any]:
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError("GGUF runtime dependency is missing. Install llama-cpp-python.") from exc
        if not self.model_path.is_file() or self.model_path.suffix.lower() != ".gguf":
            raise ValueError("Select a GGUF model file.")
        started = time.perf_counter()
        try:
            self.model = Llama(model_path=str(self.model_path), verbose=False)
        except Exception as exc:
            raise RuntimeError(f"GGUF model could not be loaded: {exc}") from exc
        self.load_seconds = time.perf_counter() - started
        return self.get_metadata()

    def unload_model(self) -> None:
        self.model = None

    def generate(self, prompt: str, max_tokens: int, temperature: float, top_p: float) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Model is not loaded.")
        result = self.model.create_completion(prompt=prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p)
        text = result["choices"][0]["text"]
        usage = result.get("usage", {})
        return {"text": text, "tokens_generated": usage.get("completion_tokens"), "finish_reason": result["choices"][0].get("finish_reason")}

    def generate_stream(self, prompt: str, max_tokens: int, temperature: float, top_p: float):
        import time
        if self.model is None:
            raise RuntimeError("Model is not loaded.")
        started = time.perf_counter()
        chunks: list[str] = []
        for item in self.model.create_completion(prompt=prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, stream=True):
            text = item["choices"][0].get("text", "")
            if text:
                chunks.append(text)
                yield {"type": "token", "text": text}
        elapsed = time.perf_counter() - started
        generated = "".join(chunks)
        tokens = len(self.model.tokenize(generated.encode("utf-8"), add_bos=False)) if generated else 0
        yield {"type": "done", "tokens_generated": tokens, "generation_time": elapsed, "tokens_per_second": tokens / elapsed if elapsed > 0 else None}

    def get_metadata(self) -> dict[str, Any]:
        details = self.model.metadata if self.model is not None else {}
        arch = details.get("general.architecture")
        file_type = details.get("general.file_type")
        quant_types = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K"}
        numeric_type = int(file_type) if isinstance(file_type, (int, str)) and str(file_type).isdigit() else file_type
        quant = quant_types.get(numeric_type, str(file_type) if file_type is not None else None)
        return {"name": details.get("general.name"), "architecture": arch, "quantization": quant, "metadata": details}

    def get_supported_features(self) -> list[str]:
        return ["load", "generate", "streaming", "metadata", "model_size", "memory"]
