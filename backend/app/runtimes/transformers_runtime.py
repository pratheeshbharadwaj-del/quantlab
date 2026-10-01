from __future__ import annotations

from importlib.util import find_spec
import time
from pathlib import Path
from typing import Any

from .base import ModelRuntime


class TransformersRuntime(ModelRuntime):
    def load_model(self) -> dict[str, Any]:
        try:
            import torch
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Transformers runtime dependencies are missing. Install torch and transformers.") from exc
        if not self.model_path.is_dir() or not (self.model_path / "config.json").is_file():
            raise RuntimeError("Transformers models must be a local directory containing config.json and tokenizer files.")
        safetensors_files = sorted(self.model_path.glob("*.safetensors"))
        if not safetensors_files and not any(self.model_path.glob("*.bin")):
            raise RuntimeError("Model weights are missing (expected *.safetensors or *.bin).")
        has_cuda = torch.cuda.is_available()
        if self.requested_device == "gpu" and not has_cuda:
            raise RuntimeError("GPU was selected, but this Python environment has no CUDA-enabled PyTorch runtime. Install a compatible CUDA build of PyTorch and reload the backend.")
        use_gpu = self.requested_device == "gpu" or (self.requested_device == "auto" and has_cuda)
        self.device = "cuda:0" if use_gpu else "cpu"
        if self.requested_device == "gpu":
            if not self._accelerate_available():
                raise RuntimeError("GPU loading for Transformers requires Accelerate. Install it with `pip install accelerate`, then reload the backend.")
        started = time.perf_counter()
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), local_files_only=True)
            load_options: dict[str, Any] = {"local_files_only": True, "torch_dtype": "auto"}
            if use_gpu and (self.requested_device == "gpu" or self._accelerate_available()):
                load_options["device_map"] = "cuda:0"
            custom_quantization = self._safetensors_quantization(safetensors_files)
            if custom_quantization == "int8-per-tensor":
                import json

                model_config = json.loads((self.model_path / "config.json").read_text(encoding="utf-8"))
                dtype_name = model_config.get("torch_dtype")
                target_dtype = getattr(torch, dtype_name, torch.float32) if isinstance(dtype_name, str) else torch.float32
                state_dict = self._load_int8_safetensors(safetensors_files, torch, target_dtype)
                config = AutoConfig.from_pretrained(str(self.model_path), local_files_only=True)
                self.model = AutoModelForCausalLM.from_config(config, dtype=target_dtype)
                self.model.load_state_dict(state_dict)
                del state_dict
                self.quantization = "INT8 per-tensor (dequantized for Transformers)"
            elif custom_quantization:
                raise RuntimeError(f"Unsupported Safetensors quantization format: {custom_quantization}.")
            else:
                self.model = AutoModelForCausalLM.from_pretrained(str(self.model_path), **load_options)
            if use_gpu and (custom_quantization == "int8-per-tensor" or "device_map" not in load_options):
                self.model.to(self.device)
            self.model.eval()
            self.torch = torch
        except Exception as exc:
            self.model = None
            raise RuntimeError(f"Local Transformers model could not be loaded: {exc}") from exc
        self.load_seconds = time.perf_counter() - started
        return self.get_metadata()

    @staticmethod
    def _safetensors_quantization(paths: list[Path]) -> str | None:
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise RuntimeError("Safetensors runtime dependency is missing. Install safetensors.") from exc
        quantization = None
        for path in paths:
            with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
                metadata = checkpoint.metadata() or {}
            shard_quantization = metadata.get("quantization")
            if shard_quantization is not None:
                if quantization is not None and quantization != shard_quantization:
                    raise RuntimeError("Safetensors shards declare inconsistent quantization formats.")
                quantization = shard_quantization
        return quantization

    @staticmethod
    def _load_int8_safetensors(paths: list[Path], torch: Any, target_dtype: Any = None) -> dict[str, Any]:
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise RuntimeError("Safetensors runtime dependency is missing. Install safetensors.") from exc
        state_dict: dict[str, Any] = {}
        suffix = ".__scale__"
        target_dtype = target_dtype or torch.float32
        for path in paths:
            with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
                keys = set(checkpoint.keys())
                scale_keys = {key for key in keys if key.endswith(suffix)}
                weight_keys = keys - scale_keys
                if not scale_keys or not weight_keys:
                    raise RuntimeError("INT8 Safetensors must store each weight with a matching .__scale__ tensor.")
                for weight_key in weight_keys:
                    scale_key = f"{weight_key}{suffix}"
                    if scale_key not in keys:
                        raise RuntimeError(f"INT8 weight {weight_key} has no matching scale tensor.")
                    weight = checkpoint.get_tensor(weight_key)
                    scale = checkpoint.get_tensor(scale_key)
                    if weight.dtype != torch.int8 or scale.numel() != 1:
                        raise RuntimeError(f"INT8 weight {weight_key} must use an int8 tensor and one per-tensor scale.")
                    scale_value = float(scale.item())
                    if not torch.isfinite(torch.tensor(scale_value)).item() or scale_value <= 0:
                        raise RuntimeError(f"INT8 weight {weight_key} has an invalid scale value.")
                    state_dict[weight_key] = weight.to(dtype=torch.float32).mul_(scale_value).to(dtype=target_dtype)
        return state_dict

    @staticmethod
    def _accelerate_available() -> bool:
        return find_spec("accelerate") is not None

    def unload_model(self) -> None:
        self.model = None
        self.tokenizer = None

    def format_messages(self, messages: list[dict[str, str]]) -> str:
        if self.model is None or not messages:
            raise RuntimeError("A loaded model and at least one chat message are required.")
        template = getattr(self.tokenizer, "chat_template", None)
        if template:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
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
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                **({"temperature": temperature, "top_p": top_p} if temperature > 0 else {}),
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        return {"text": self.tokenizer.decode(new_tokens, skip_special_tokens=True), "tokens_generated": int(new_tokens.shape[0]), "finish_reason": "length" if new_tokens.shape[0] >= max_tokens else "eos"}

    def get_metadata(self) -> dict[str, Any]:
        config = getattr(self.model, "config", None)
        return {"architecture": getattr(config, "architectures", [None])[0] if config else None, "parameter_count": sum(p.numel() for p in self.model.parameters()) if self.model is not None else None, "quantization": getattr(self, "quantization", None), "device": str(next(self.model.parameters()).device) if self.model is not None else None, "chat_template_available": bool(getattr(self.tokenizer, "chat_template", None)) if self.model is not None else False}

    def get_supported_features(self) -> list[str]:
        return ["load", "generate", "metadata", "model_size", "memory", "perplexity"]

    def perplexity(self, text: str) -> float | None:
        if self.model is None or not text.strip():
            return None
        try:
            encoded = self.tokenizer(text, return_tensors="pt")
            encoded = {name: tensor.to(self.device) for name, tensor in encoded.items()}
            with self.torch.inference_mode():
                loss = self.model(**encoded, labels=encoded["input_ids"]).loss
            return float(self.torch.exp(loss).item())
        except Exception:
            return None
