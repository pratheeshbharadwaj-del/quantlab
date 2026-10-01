from __future__ import annotations

import csv
import io
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Literal

import psutil
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.evaluation.metrics import output_similarity, score_outputs
from app.services.hardware_monitor import system_status
from app.services.model_manager import ModelManager

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_ROOT = Path(os.environ.get("QNT_MODELS_DIR", BASE_DIR / "models")).resolve()
MODEL_ROOT.mkdir(parents=True, exist_ok=True)
manager = ModelManager(MODEL_ROOT)
evaluations: dict[str, dict[str, Any]] = {}
DATASET_PATH = BASE_DIR / "evaluation_dataset.json"
MAX_MODEL_UPLOAD_BYTES = 16 * 1024**3
MODEL_UPLOAD_EXTENSIONS = {".zip", ".gguf", ".onnx", ".tflite", ".lite", ".safetensors", ".bin", ".pt", ".pth"}
GENERATION_MAX_TOKENS = 64
GENERATION_TEMPERATURE = 0.6
GENERATION_TOP_P = 0.95

DEFAULT_ALLOWED_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
_extra_origins = [origin.strip() for origin in os.environ.get("QNT_ALLOWED_ORIGINS", "").split(",") if origin.strip()]
ALLOWED_ORIGINS = _extra_origins or DEFAULT_ALLOWED_ORIGINS

app = FastAPI(title="Local Model Evaluation", version="1.0.0", description="Local-only model inference and quantization evaluation.")
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["*"], allow_headers=["*"])


class ImportRequest(BaseModel):
    path: str = Field(min_length=1, max_length=500)
    format_hint: str = "AUTO"
    role: str = "library"


class GenerateRequest(BaseModel):
    model_id: str
    prompt: str = Field(default="", max_length=100_000)
    messages: list[dict[str, str]] | None = None


class EvalRequest(BaseModel):
    model_id: str
    dataset: list[dict[str, Any]] | None = None


class CompareRequest(BaseModel):
    original_model_id: str
    quantized_model_id: str
    dataset: list[dict[str, Any]] | None = None
    prompt: str | None = None


class BenchmarkRequest(BaseModel):
    model_id: str
    prompt: str = Field(min_length=1, max_length=100_000)
    runs: int = Field(default=3, ge=1, le=10)


class DeviceRequest(BaseModel):
    device: Literal["auto", "cpu", "gpu"]


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, KeyError):
        return HTTPException(404, str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(400, str(exc))
    if isinstance(exc, RuntimeError):
        return HTTPException(422, str(exc))
    return HTTPException(500, f"Operation failed: {exc}")


def _dataset(custom: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    data = custom if custom is not None else json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    if not data or any(not isinstance(row.get("question"), str) or not row["question"].strip() for row in data):
        raise ValueError("Evaluation dataset must contain non-empty question fields.")
    return data


def _peak_process_rss() -> int | None:
    info = psutil.Process().memory_info()
    peak = getattr(info, "peak_wset", None)
    return int(peak) if peak is not None else None


def _evaluate_one(model_id: str, dataset: list[dict[str, Any]]) -> dict[str, Any]:
    results = []
    runtime = manager.loaded[model_id]
    formatter = getattr(runtime, "format_messages", None)
    for row in dataset:
        prompt = formatter([{"role": "user", "content": row["question"]}]) if callable(formatter) else row["question"]
        generated, _ = manager.generate(model_id, prompt, GENERATION_MAX_TOKENS, GENERATION_TEMPERATURE, GENERATION_TOP_P)
        results.append({"id": row.get("id"), "question": row["question"], "expected_answer": row.get("expected_answer"), "output": generated["text"], "generation_time": generated["generation_time"], "tokens_generated": generated["tokens_generated"], "tokens_per_second": generated["tokens_per_second"]})
    scores = score_outputs(results)
    refs = "\n".join(str(row.get("expected_answer", "")) for row in dataset)
    ppl = runtime.perplexity(refs) if refs.strip() else None
    times = [row["generation_time"] for row in results]
    return {"model_id": model_id, "model_name": manager.get(model_id)["name"], "sample_count": len(results), "metrics": scores, "perplexity": ppl, "average_generation_time": sum(times) / len(times), "results": results}


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "mode": "LOCAL MODE", "model_directory": str(MODEL_ROOT)}


@app.post("/system/device")
def set_inference_device(request: DeviceRequest) -> dict[str, str]:
    try:
        return {"inference_device": manager.set_device_preference(request.device)}
    except Exception as exc:
        raise _http_error(exc) from exc


@app.get("/system/status")
def get_system_status() -> dict[str, Any]:
    return {**system_status(), "inference_device": manager.device_preference}


@app.post("/models/import")
def import_model(request: ImportRequest) -> dict[str, Any]:
    try:
        if request.role not in {"original", "quantized", "library"}:
            raise ValueError("Model role must be original, quantized, or library.")
        expected = {"GGUF": {"GGUF"}, "TRANSFORMERS": {"SAFETENSORS", "BIN"}, "SAFETENSORS": {"SAFETENSORS"}, "PYTORCH": {"PT", "PTH"}, "ONNX": {"ONNX"}, "TFLITE": {"TFLITE", "LITE"}}
        hint = request.format_hint.upper()
        if hint not in {"AUTO", *expected}:
            raise ValueError("Unknown format selection.")
        record = manager.import_model(request.path)
        if hint != "AUTO" and record["format"] not in expected[hint]:
            raise ValueError(f"Selected format {hint} does not match detected format {record['format']}.")
        return _assign_model_role(record, request.role)
    except Exception as exc:
        raise _http_error(exc) from exc


def _assign_model_role(record: dict[str, Any], role: str) -> dict[str, Any]:
    record["role"] = role
    if role == "quantized" and record.get("supported"):
        try:
            loaded_record = manager.load(record["id"])
            loaded_record["role"] = role
            return loaded_record
        except Exception as exc:
            record["loaded"] = False
            record["auto_load_error"] = str(exc)
    return record


@app.post("/models/upload")
async def upload_model(file: UploadFile = File(...), role: str = Form("original")) -> dict[str, Any]:
    if role not in {"original", "quantized"}:
        raise HTTPException(400, "Model role must be original or quantized.")
    original_name = (file.filename or "model").replace("\\", "/").rsplit("/", 1)[-1]
    extension = Path(original_name).suffix.lower()
    if extension not in MODEL_UPLOAD_EXTENSIONS:
        raise HTTPException(400, f"Unsupported model file extension: {extension or 'none'}. Upload a model archive or supported model file.")
    upload_dir = (MODEL_ROOT / ".uploads").resolve()
    if MODEL_ROOT not in upload_dir.parents:
        raise HTTPException(400, "Invalid local upload directory.")
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(original_name).stem).strip("._") or "model"
    destination = upload_dir / f"{uuid.uuid4().hex}_{safe_name}{extension}"
    total = 0
    try:
        with destination.open("wb") as output:
            while chunk := await file.read(8 * 1024 * 1024):
                total += len(chunk)
                if total > MAX_MODEL_UPLOAD_BYTES:
                    raise HTTPException(413, "Model upload exceeds the 16 GiB limit.")
                output.write(chunk)
        if total == 0:
            raise HTTPException(400, "The selected model file is empty.")
        relative_path = str(destination.relative_to(MODEL_ROOT))
        record = manager.import_model(relative_path)
        record["name"] = Path(original_name).stem
        return _assign_model_role(record, role)
    except HTTPException:
        destination.unlink(missing_ok=True)
        raise
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise _http_error(exc) from exc
    finally:
        await file.close()


@app.post("/models/load")
def load_model(payload: dict[str, str]) -> dict[str, Any]:
    try:
        model_id = payload.get("model_id")
        if not model_id and payload.get("path"):
            record = manager.import_model(payload["path"])
            model_id = record["id"]
        if not model_id:
            raise ValueError("Provide model_id or a path inside the configured models directory.")
        return manager.load(model_id)
    except Exception as exc:
        raise _http_error(exc) from exc


@app.post("/models/unload")
def unload_model(payload: dict[str, str] | None = None) -> dict[str, Any]:
    try:
        return manager.unload((payload or {}).get("model_id"))
    except Exception as exc:
        raise _http_error(exc) from exc


@app.get("/models")
def list_models() -> list[dict[str, Any]]:
    return manager.list_models()


@app.get("/models/files")
def list_model_files(path: str = "") -> list[dict[str, Any]]:
    try:
        return manager.list_files(path)
    except Exception as exc:
        raise _http_error(exc) from exc


@app.get("/models/{model_id}")
def get_model(model_id: str) -> dict[str, Any]:
    try:
        record = manager.get(model_id)
        return {**record, "loaded": model_id in manager.loaded}
    except Exception as exc:
        raise _http_error(exc) from exc


@app.get("/models/{model_id}/info")
def get_model_info(model_id: str) -> dict[str, Any]:
    return get_model(model_id)


@app.post("/generate")
def generate(request: GenerateRequest) -> dict[str, Any]:
    try:
        runtime = manager.loaded.get(request.model_id)
        if runtime is None:
            raise RuntimeError("Load this model before generating text.")
        prompt = _generation_prompt(runtime, request.prompt, request.messages)
        output, _ = manager.generate(request.model_id, prompt, GENERATION_MAX_TOKENS, GENERATION_TEMPERATURE, GENERATION_TOP_P)
        return {"model_id": request.model_id, **output}
    except Exception as exc:
        raise _http_error(exc) from exc


@app.post("/generate/stream")
def generate_stream(request: GenerateRequest) -> StreamingResponse:
    runtime = manager.loaded.get(request.model_id)
    if runtime is None:
        raise HTTPException(422, "Load this model before generating text.")
    try:
        prompt = _generation_prompt(runtime, request.prompt, request.messages)
    except Exception as exc:
        raise _http_error(exc) from exc
    def events():
        stream = runtime.generate_stream(prompt, GENERATION_MAX_TOKENS, GENERATION_TEMPERATURE, GENERATION_TOP_P)
        try:
            for event in stream:
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)}, ensure_ascii=False)}\n\n"
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                close()
    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _generation_prompt(runtime: Any, prompt: str, messages: list[dict[str, str]] | None) -> str:
    if messages:
        formatter = getattr(runtime, "format_messages", None)
        if callable(formatter):
            return formatter(messages)
        conversation = "\n".join(f"{message.get('role', 'user').title()}: {message.get('content', '')}" for message in messages)
        return f"{conversation}\nAssistant:"
    if not prompt.strip():
        raise ValueError("Provide a prompt or chat messages.")
    return prompt


@app.post("/evaluation/run")
def run_evaluation(request: EvalRequest) -> dict[str, Any]:
    try:
        result = _evaluate_one(request.model_id, _dataset(request.dataset))
        evaluation_id = str(uuid.uuid4())
        result["id"] = evaluation_id
        evaluations[evaluation_id] = result
        return result
    except Exception as exc:
        raise _http_error(exc) from exc


@app.post("/evaluation/compare")
def compare(request: CompareRequest) -> dict[str, Any]:
    try:
        if request.original_model_id == request.quantized_model_id:
            raise ValueError("Choose two different models for comparison.")
        original = manager.get(request.original_model_id)
        quantized = manager.get(request.quantized_model_id)
        if original.get("content_fingerprint") and original.get("content_fingerprint") == quantized.get("content_fingerprint"):
            raise ValueError("These imports contain identical model weights. Upload the actual original pretrained model and its quantized version as separate files.")
        for model_id, record in ((request.original_model_id, original), (request.quantized_model_id, quantized)):
            if model_id not in manager.loaded:
                raise RuntimeError(f"Load {record['name']} before starting the comparison.")
            if not record.get("generation_supported"):
                raise RuntimeError(f"Text generation is not supported for {record['name']} on its selected runtime.")
        dataset = _dataset(request.dataset) if request.dataset is not None or not request.prompt else []
        original_result = _evaluate_one(request.original_model_id, dataset) if dataset else None
        quantized_result = _evaluate_one(request.quantized_model_id, dataset) if dataset else None
        prompt_result = None
        if request.prompt:
            left, _ = manager.generate(request.original_model_id, request.prompt, GENERATION_MAX_TOKENS, GENERATION_TEMPERATURE, GENERATION_TOP_P)
            right, _ = manager.generate(request.quantized_model_id, request.prompt, GENERATION_MAX_TOKENS, GENERATION_TEMPERATURE, GENERATION_TOP_P)
            prompt_result = {"prompt": request.prompt, "original_output": left["text"], "quantized_output": right["text"], "output_similarity": output_similarity(left["text"], right["text"]), "original_generation": left, "quantized_generation": right}
        original_size, quantized_size = original["size_bytes"], quantized["size_bytes"]
        comparison = {
            "original": {"model": original, "evaluation": original_result},
            "quantized": {"model": quantized, "evaluation": quantized_result},
            "size_reduction_percent": (original_size - quantized_size) / original_size * 100 if original_size else None,
            "memory": {"original_ram_bytes": manager.loaded[request.original_model_id].ram_increase_bytes if hasattr(manager.loaded[request.original_model_id], "ram_increase_bytes") else None, "quantized_ram_bytes": manager.loaded[request.quantized_model_id].ram_increase_bytes if hasattr(manager.loaded[request.quantized_model_id], "ram_increase_bytes") else None, "original_before_bytes": manager.loaded[request.original_model_id].ram_before_load_bytes if hasattr(manager.loaded[request.original_model_id], "ram_before_load_bytes") else None, "original_after_bytes": manager.loaded[request.original_model_id].ram_after_load_bytes if hasattr(manager.loaded[request.original_model_id], "ram_after_load_bytes") else None, "quantized_before_bytes": manager.loaded[request.quantized_model_id].ram_before_load_bytes if hasattr(manager.loaded[request.quantized_model_id], "ram_before_load_bytes") else None, "quantized_after_bytes": manager.loaded[request.quantized_model_id].ram_after_load_bytes if hasattr(manager.loaded[request.quantized_model_id], "ram_after_load_bytes") else None},
            "prompt_comparison": prompt_result,
            "interpretation": "Exact match and token F1 are lexical comparisons against the supplied reference answers; they do not determine semantic correctness. Accuracy is unavailable for free-form text without human labels or a validated task-specific scorer. Output similarity is lexical Jaccard overlap, not accuracy. N/A values mean the selected runtime does not support that measurement."
        }
        comparison["accuracy_difference"] = None
        comparison["token_f1_difference"] = (quantized_result["metrics"]["token_f1"] - original_result["metrics"]["token_f1"]) if original_result and quantized_result and original_result["metrics"]["token_f1"] is not None and quantized_result["metrics"]["token_f1"] is not None else None
        evaluation_id = str(uuid.uuid4())
        comparison["id"] = evaluation_id
        evaluations[evaluation_id] = comparison
        return comparison
    except Exception as exc:
        raise _http_error(exc) from exc


@app.get("/evaluation/dataset")
def get_dataset() -> list[dict[str, Any]]:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


@app.get("/evaluation/{evaluation_id}")
def get_evaluation(evaluation_id: str) -> dict[str, Any]:
    if evaluation_id not in evaluations:
        raise HTTPException(404, "Evaluation result not found.")
    return evaluations[evaluation_id]


@app.post("/evaluation/dataset/import")
async def import_dataset(file: UploadFile = File(...)) -> dict[str, Any]:
    if Path(file.filename or "").suffix.lower() not in {".json", ".csv"}:
        raise HTTPException(400, "Upload a JSON or CSV dataset.")
    raw = await file.read(5 * 1024 * 1024 + 1)
    if len(raw) > 5 * 1024 * 1024:
        raise HTTPException(413, "Dataset file exceeds 5 MB.")
    try:
        if Path(file.filename or "").suffix.lower() == ".json":
            data = json.loads(raw.decode("utf-8"))
        else:
            data = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
        valid = _dataset(data)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(400, f"Invalid dataset: {exc}") from exc
    return {"dataset": valid, "count": len(valid), "source": "uploaded file; retained in the browser session only"}


@app.post("/benchmark/run")
def run_benchmark(request: BenchmarkRequest) -> dict[str, Any]:
    try:
        runtime = manager.loaded.get(request.model_id)
        if runtime is None:
            raise RuntimeError("Load this model before running a benchmark.")
        records = []
        for index in range(request.runs):
            start_rss = psutil.Process().memory_info().rss
            started = time.perf_counter()
            output, _ = manager.generate(request.model_id, request.prompt, GENERATION_MAX_TOKENS, GENERATION_TEMPERATURE, GENERATION_TOP_P)
            records.append({"run": index + 1, "generation_time": time.perf_counter() - started, "first_token_latency_seconds": None, "tokens_generated": output["tokens_generated"], "tokens_per_second": output["tokens_per_second"], "process_rss_before_bytes": start_rss, "process_rss_after_bytes": psutil.Process().memory_info().rss, "peak_process_rss_bytes": _peak_process_rss(), "warmup": index == 0 and request.runs > 1})
        measured = [r for r in records if not r["warmup"]] or records
        speeds = [r["tokens_per_second"] for r in measured if r["tokens_per_second"] is not None]
        times = [r["generation_time"] for r in measured]
        return {"model_id": request.model_id, "runs": records, "warmup_discarded": len(measured) != len(records), "summary": {"generation_time_average": sum(times) / len(times), "generation_time_min": min(times), "generation_time_max": max(times), "tokens_per_second_average": sum(speeds) / len(speeds) if speeds else None, "tokens_per_second_min": min(speeds) if speeds else None, "tokens_per_second_max": max(speeds) if speeds else None}}
    except Exception as exc:
        raise _http_error(exc) from exc
