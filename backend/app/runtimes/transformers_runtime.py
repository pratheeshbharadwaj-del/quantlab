from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .base import ModelRuntime


class TransformersRuntime(ModelRuntime):
    def load_model(self) -> dict[str, Any]:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Transformers runtime dependencies are missing. Install torch and transformers.") from exc
        if not self.model_path.is_dir() or not (self.model_path / "config.json").is_file():
            raise RuntimeError("Transformers models must be a local directory containing config.json and tokenizer files.")
        if not any(self.model_path.glob("*.safetensors")) and not any(self.model_path.glob("*.bin")):
            raise RuntimeError("Model weights are missing (expected *.safetensors or *.bin).")
        started = time.perf_counter()
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), local_files_only=True)
            self.model = AutoModelForCausalLM.from_pretrained(str(self.model_path), local_files_only=True, torch_dtype="auto")
            self.model.eval()
            self.torch = torch
        except Exception as exc:
            self.model = None
            raise RuntimeError(f"Local Transformers model could not be loaded: {exc}") from exc
        self.load_seconds = time.perf_counter() - started
        return self.get_metadata()

    def unload_model(self) -> None:
        self.model = None
        self.tokenizer = None

    def format_messages(self, messages: list[dict[str, str]]) -> str:
        if self.model is None or not messages:
            raise RuntimeError("A loaded model and at least one chat message are required.")
        template = getattr(self.tokenizer, "chat_template", None)
        if template:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        last_user_message = next((item["content"] for item in reversed(messages) if item.get("role") == "user"), "")
        if not last_user_message:
            raise ValueError("Chat input must include a user message.")
        model_type = getattr(getattr(self.model, "config", None), "model_type", "")
        if model_type == "gpt2":
            # Base GPT-2 checkpoints are completion models, not role-conditioned chat models.
            return f"Question: {last_user_message.strip()}\nAnswer:"
        return f"User: {last_user_message.strip()}\nAssistant:"

    def generate(self, prompt: str, max_tokens: int, temperature: float, top_p: float) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Model is not loaded.")
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                repetition_penalty=1.15,
            )
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        return {"text": self.tokenizer.decode(new_tokens, skip_special_tokens=True), "tokens_generated": int(new_tokens.shape[0]), "finish_reason": "length" if new_tokens.shape[0] >= max_tokens else "eos"}

    def get_metadata(self) -> dict[str, Any]:
        config = getattr(self.model, "config", None)
        return {"architecture": getattr(config, "architectures", [None])[0] if config else None, "parameter_count": sum(p.numel() for p in self.model.parameters()) if self.model is not None else None, "quantization": None, "device": str(next(self.model.parameters()).device) if self.model is not None else None, "chat_template_available": bool(getattr(self.tokenizer, "chat_template", None)) if self.model is not None else False}

    def get_supported_features(self) -> list[str]:
        return ["load", "generate", "metadata", "model_size", "memory", "perplexity"]

    def perplexity(self, text: str) -> float | None:
        if self.model is None or not text.strip():
            return None
        try:
            encoded = self.tokenizer(text, return_tensors="pt")
            with self.torch.inference_mode():
                loss = self.model(**encoded, labels=encoded["input_ids"]).loss
            return float(self.torch.exp(loss).item())
        except Exception:
            return None
