# QuantLab — AI Model Quantization & Efficient Local Inference

QuantLab is a local-first academic demonstration app for importing, running, and evaluating pretrained models against quantized variants. It does **not** train models or quantize weights. No login, API keys, cloud inference, or automatic model downloads are used.

All quality and performance figures are measured from real runs. Unsupported metrics are returned as `N/A`/`null`; no benchmark values are synthesized.

## Features

- FastAPI backend and React/TypeScript interface.
- Local model registry restricted to a configured model directory, with safe local ZIP extraction for packaged model folders.
- Format inspection for GGUF, local Hugging Face Transformers/Safetensors, ONNX, TensorFlow Lite, and PyTorch tensor checkpoints.
- GGUF generation through `llama-cpp-python`; local causal-LM generation through Transformers/PyTorch.
- Chat, same-prompt original-vs-quantized comparison, custom JSON/CSV evaluation datasets, and repeatable inference benchmarks.
- Project Evaluation can upload an original model locally (manual load) and upload a quantized model that auto-loads when the runtime accepts it.
- Actual on-disk bytes, load duration, process RSS before/after loading, generation latency, runtime token count/throughput when available, and hardware status.
- Built-in 20-question evaluation dataset.
- Path traversal prevention and safe weights-only PyTorch checkpoint inspection.

## Architecture

- `backend/app/main.py`: REST API and evaluation orchestration.
- `backend/app/runtimes/`: `ModelRuntime` contract and GGUF, Transformers, ONNX, TFLite, and PyTorch adapters.
- `backend/app/services/`: local model discovery, runtime selection, and hardware monitoring.
- `backend/app/evaluation/metrics.py`: exact match, token F1, and lexical output similarity.
- `backend/evaluation_dataset.json`: local sample dataset.
- `frontend/src/`: responsive dashboard, model library/import, local chat, Project Evaluation, benchmark, and settings pages.

## Supported formats and honest limits

| Input | Adapter | What works |
|---|---|---|
| `.gguf` | `llama-cpp-python` | Metadata inspection and local text generation; token streaming when supported by the installed llama.cpp runtime. Quantization is read from GGUF metadata after load. |
| Local Transformers directory or ZIP with `config.json`, tokenizer files, and weights (`*.safetensors` or `*.bin`) | Transformers + PyTorch | Local causal-language-model loading and generation; perplexity where the architecture supports loss calculation. `local_files_only=True` prevents downloads. ZIP extraction is path-checked and limited to 32 GiB uncompressed. |
| `.onnx` | ONNX Runtime | Safe session load and metadata inspection. Generic ONNX files do not define a universal text-generation/tokenizer interface, so generation is explicitly unavailable in this adapter. |
| `.tflite` / `.lite` | TensorFlow Lite / LiteRT | Interpreter load and tensor metadata inspection. Generic TFLite graphs do not define a universal text-generation interface. |
| `.pt` / `.pth` | PyTorch | Weights-only checkpoint inspection with `weights_only=True`. A state dict does not identify the architecture/tokenizer, so generic text generation is not claimed. |

A recognized format is not a guarantee that every architecture will load. Missing dependencies, invalid files, unsupported architectures, or missing tokenizer files produce a descriptive error instead of a fabricated result. Optional adapters can be extended independently of the frontend.

## Requirements

- Windows 10/11, macOS, or Linux.
- Python 3.10+ (the current workspace was configured with Python 3.14).
- Node.js 20+ and npm.
- Enough system RAM/storage for the selected model. Large models may need a compatible GPU/runtime, but CPU inference is runtime/model dependent.

## Installation

### Backend

From the project root, create and activate a virtual environment, then install the backend dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
```

For backend tests, install the development dependencies instead:

```powershell
pip install -r backend\requirements-dev.txt
```

Install only the optional runtime packages needed for your models, using the PyTorch/accelerator build appropriate to your machine:

```powershell
pip install llama-cpp-python
pip install torch transformers safetensors accelerate
pip install bitsandbytes  # only for models quantized with bitsandbytes
pip install onnxruntime
pip install ai-edge-litert
```

The TensorFlow Lite adapter also accepts a TensorFlow installation when LiteRT is unavailable. Optional inference runtimes can have platform/GPU-specific installation requirements; Transformers requires Accelerate for bitsandbytes 4-bit loading, and bitsandbytes execution depends on a supported bitsandbytes build and device backend. QuantLab reports required missing packages and verifies actual load compatibility rather than estimating it. This workspace's CPU-only Windows environment loaded and generated from the provided GPT-2 NF4 model after installing these packages; compatibility may differ for other models/devices.

Start the API **from the backend directory** after activating the environment:

```powershell
cd backend
python -m uvicorn app.main:app --reload
```

API documentation is available at `http://127.0.0.1:8000/docs` while the backend is running.

### Frontend

In another terminal, from the project root:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`. The UI connects to `http://127.0.0.1:8000` by default. Set `VITE_API_URL` at build/dev time only if the local backend uses another address.

## Import a local model

1. Copy/place the files or ZIP archive under `backend/models/` (or set `QNT_MODELS_DIR` to another trusted local directory before starting the backend).
2. Open **Models → Import model**.
3. Enter a relative path, such as `my-model.gguf`, `original-model/`, or the archive filename `D__qnt.zip`. ZIP archives are safely extracted locally before model inspection.
4. Keep **Automatic detection** selected unless you want to assert a format and have the backend validate the assertion.
5. Inspect detected format, architecture, file size, and runtime. Select **Load model** only if the adapter is supported and its dependency is installed.

Model files are read in place, not uploaded to the web page or copied to an external service. Local Transformers models must already contain their configuration, tokenizer, and weights; QuantLab does not download them.

## Run inference

Load a compatible GGUF or Transformers causal language model, open **Inference lab**, select the loaded model, and send a prompt. Temperature, top-p, system prompt, and token-limit controls are intentionally removed; the backend applies the same deterministic fixed generation profile (maximum 64 generated tokens) for repeatable results. GGUF responses stream when supported; other adapters may return the complete response after generation. The stop action closes the active streaming request; interruption behavior for non-streaming runtime calls depends on that runtime. Chat generation time and token rate are reported from completed runtime output only.

## Compare original and quantized models

1. Import two distinct variants and load each (both need to fit available memory simultaneously for this in-process comparison). QuantLab rejects matching weight fingerprints, which usually means the same model was uploaded in both slots.
2. Open **Project Evaluation** and choose one model for each side.
3. Keep the built-in dataset or import a JSON array / CSV with `question` and optional `expected_answer` fields.
4. Optionally provide a same prompt and run the comparison.
5. Review actual file sizes, measured process RSS change at each load, generated outputs, Exact Match, token F1, output similarity, tokens/sec, and perplexity if available.

From the Project Evaluation comparison panel, use **Upload original locally** to import an original model file/archive, then choose **Load original model**. Use **Import quantized · auto-load** for the quantized file; after import the backend loads it automatically when its runtime is available. These uploads go only to the local backend and are capped at 16 GiB per file. A Transformers model with multiple files should be uploaded as a ZIP archive containing its config, tokenizer, and weights.

Metrics:

- **Exact Match**: normalized generated text exactly equals its expected answer. This is a strict dataset metric, not a universal model-accuracy claim.
- **Token F1**: F1 of normalized word-token overlap against the supplied expected answer.
- **Output Similarity**: Jaccard overlap of unique normalized output words for the same prompt. It is not accuracy or semantic embedding similarity.
- **Perplexity**: computed from expected evaluation text only when the loaded Transformers causal-LM adapter supports loss calculation. Lower perplexity generally indicates better prediction of that text; compare using the same text.
- **Size reduction**: calculated from actual total file sizes as `(original bytes - quantized bytes) / original bytes × 100`.
- **RAM**: process RSS immediately before and after loading; the difference is a measured process-level increase, not an isolated per-model allocation. GPU memory is reported only when the available PyTorch CUDA interface can observe it.

## Run benchmarks

Open **Benchmarks**, choose a loaded text-generation model, set one of 1/3/5/10 runs, and enter a prompt. One run is selected by default so results appear quickly; the first run is excluded as warm-up when more than one run is selected. The backend applies the same fixed 64-token generation ceiling. The page reports measured wall time and tokens/sec only when the adapter returns an actual token count. First-token latency is currently unavailable and returned as `null`; peak process RSS is returned only on platforms exposing a high-water mark.

For a fair original/quantized performance test, use the same prompt, generation settings, model context, and run count; avoid unrelated system load.

## Evaluation API

- `POST /models/import`, `POST /models/upload`, `POST /models/load`, `POST /models/unload`
- `GET /models`, `GET /models/{id}`, `GET /models/{id}/info`
- `POST /generate`, `POST /generate/stream`
- `POST /evaluation/run`, `POST /evaluation/compare`, `GET /evaluation/{id}`
- `GET /evaluation/dataset`, `POST /evaluation/dataset/import`
- `POST /benchmark/run`
- `GET /system/status`

The API explorer at `/docs` lists schemas and payloads.

## Tests and production build

Run backend tests from the project root:

```powershell
.venv\Scripts\python.exe -c "import sys; sys.path.insert(0, 'backend'); import pytest; raise SystemExit(pytest.main(['-q', 'backend/tests']))"
```

Build the frontend:

```powershell
cd frontend
npm run build
```

## Known limitations

- QuantLab runs inference/evaluation only. It does not train or quantize models.
- Generic ONNX and TFLite model graphs are inspectable/loadable, but text generation requires architecture-specific tokenizer/generation logic not defined by those file formats. The current adapters do not pretend otherwise.
- Standalone PyTorch checkpoints are inspected in safe weights-only mode; no arbitrary pickle-based model execution is used.
- Only local causal language models through GGUF and Transformers have generic text-generation support in this version. Model-specific chat templates are not automatically inferred for all architectures.
- Exact Match and token F1 depend on dataset wording and are not interchangeable with benchmark accuracy. Lexical output similarity does not imply semantic equivalence.
- RAM load increase uses process RSS and may include other allocations; GPU memory is not attributed per model. Peak RSS/first-token latency may be unavailable by OS/runtime.
- Streaming cancellation is effective when the runtime stream can be closed (GGUF); a non-streaming generation call may continue within its runtime after a client disconnect.
- This local demo keeps imported model/evaluation records in memory; restarting the backend clears the registry and saved evaluation IDs. Model files and built-in dataset remain on disk.
- There is no built-in upload of multi-gigabyte model weights. Place model files inside the allowed local model directory before importing.
