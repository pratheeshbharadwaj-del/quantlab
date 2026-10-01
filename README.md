# QuantLab — AI Model Quantization & Efficient Local Inference

QuantLab is a local-first academic demonstration app for importing, running, and evaluating pretrained models against quantized variants. It does **not** train models or quantize weights. No login, API keys, cloud inference, or automatic model downloads are used — "local-first" refers to *inference*: all model loading and generation always happens on whichever machine runs the backend process, never via a third-party AI API. That backend can be run on your laptop (default) or hosted on a server/cloud VM so a team can share one instance (see [Cloud deployment](#cloud-deployment)); the inference guarantee is unchanged either way.

All quality and performance figures are measured from real runs. Unsupported metrics are returned as `N/A`/`null`; no benchmark values are synthesized.

For free-form answer evaluations, **Exact Match** and **Token F1** are text-overlap measures only; they do not judge meaning. Semantic accuracy is unavailable unless the task supplies suitable labels and a validated scorer.

## Features

- FastAPI backend and React/TypeScript interface.
- Local model registry restricted to a configured model directory, with safe local ZIP extraction for packaged model folders.
- Format inspection for GGUF, local Hugging Face Transformers/Safetensors, ONNX, TensorFlow Lite, and PyTorch tensor checkpoints.
- GGUF generation through `llama-cpp-python`; local causal-LM generation through Transformers/PyTorch.
- Chat, same-prompt original-vs-quantized comparison, custom JSON/CSV evaluation datasets, and repeatable inference benchmarks.
- Project Evaluation can upload an original model locally (manual load) and upload a quantized model that auto-loads when the runtime accepts it.
- Actual on-disk bytes, load duration, process RSS before/after loading, generation latency, runtime token count/throughput when available, and hardware status.
- Select Auto, CPU, or GPU for subsequent model loads. Explicit GPU mode fails clearly when the installed adapter cannot use a GPU rather than silently running inference on the CPU.
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
| `.gguf` | `llama-cpp-python` | Metadata inspection and local text generation; Auto uses GPU offload when the installed build supports it, and explicit GPU mode requires a GPU-enabled build. Token streaming is available when supported. |
| Local Transformers directory or ZIP with `config.json`, tokenizer files, and weights (`*.safetensors` or `*.bin`) | Transformers + PyTorch | Local causal-language-model loading and generation; Auto uses CUDA when available, CPU can be forced, and explicit GPU mode requires CUDA-enabled PyTorch and Accelerate. `local_files_only=True` prevents downloads. Custom `int8-per-tensor` Safetensors use their stored scales and are dequantized to the model dtype at load time; this is not native low-bit inference and can use substantially more RAM. ZIP extraction is path-checked and limited to 32 GiB uncompressed. |
| `.onnx` | ONNX Runtime | Safe session load and metadata inspection. Auto/GPU use CUDAExecutionProvider when installed; explicit GPU mode errors if it is unavailable. Generic ONNX files do not define a universal text-generation/tokenizer interface. |
| `.tflite` / `.lite` | TensorFlow Lite / LiteRT | CPU interpreter load and tensor metadata inspection. GPU selection is explicitly unsupported in this adapter. Generic TFLite graphs do not define a universal text-generation interface. |
| `.pt` / `.pth` | PyTorch | Weights-only checkpoint inspection with `weights_only=True`, mapped to the selected CPU/GPU device. A state dict does not identify the architecture/tokenizer, so generic text generation is not claimed. |

Set the inference preference on **Settings & hardware** before loading a model. **Auto** chooses a GPU only when the selected model's installed runtime can use one; otherwise it uses CPU. **GPU** never silently falls back to CPU, and **CPU** forces CPU execution. The selector reports GPU-capable installed adapters. Changing the preference does not move models already in memory: unload and reload them to apply it. GPU support depends on compatible accelerator-enabled builds (for example, CUDA-enabled PyTorch plus Accelerate for Transformers or a CUDA-enabled `llama-cpp-python` build for GGUF). QuantLab does not install runtimes or download models automatically.

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
5. Review actual file sizes, measured process RSS change at each load, generated outputs, Exact Match and Token F1 text-overlap scores, output similarity, tokens/sec, and perplexity if available.

The comparison view displays **Reference overlap (Token F1)** as its prominent side-by-side score, so answers with shared wording receive partial credit instead of both looking like 0%. **Exact Match** is shown as a secondary strict text check; it requires normalized text to match exactly. Neither measure evaluates meaning, so 0% Exact Match does not necessarily mean an answer is incorrect. The app reports semantic accuracy as unavailable for free-form answers unless a validated task-specific scorer is supplied. The separate "Measured change" panel shows measured size, RAM, and speed deltas.

From the Project Evaluation comparison panel, use **Upload original locally** to import an original model file/archive, then choose **Load original model**. Use **Import quantized · auto-load** for the quantized file; after import the backend loads it automatically when its runtime is available. These uploads go only to the local backend and are capped at 16 GiB per file. A Transformers model with multiple files should be uploaded as a ZIP archive containing its config, tokenizer, and weights.

Metrics:

- **Accuracy**: unavailable for free-form text without human labels or a validated task-specific scorer. QuantLab does not infer semantic correctness from word overlap.
- **Exact Match**: normalized generated text exactly equals its expected answer. This is a lexical check, not semantic correctness.
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

## Cloud deployment

QuantLab can be hosted on a server or in the cloud so multiple people share one instance. Inference still runs entirely on that host — no third-party AI API is used, and models are still loaded only from the configured local directory. There is still **no** automatic model download; whoever hosts the instance must place model files in the mounted model directory themselves.

### Environment variables

| Variable | Where | Purpose | Default |
|---|---|---|---|
| `QNT_MODELS_DIR` | backend | Root directory the backend reads/writes model files from. | `./models` |
| `QNT_ALLOWED_ORIGINS` | backend | Comma-separated list of origins allowed to call the API (CORS). Replaces, not appends to, the built-in localhost list. | `http://localhost:5173,http://127.0.0.1:5173` |
| `QNT_API_URL` | frontend (container only) | Backend base URL the served frontend should call. Injected into `config.js` at container **start** (not build time), so one built image can point at any backend without rebuilding. | *(same-origin/empty → `http://127.0.0.1:8000` in dev)* |

### Option A — Docker Compose (single server, simplest)

Run both services on any machine with Docker installed:

```powershell
docker compose up --build -d
```

This starts the backend on `http://localhost:8000` and the frontend on `http://localhost:8080`, with model files persisted in a named Docker volume (`quantlab-models`) so they survive container restarts. Override ports/origins via a `.env` file or shell environment variables (`QNT_ALLOWED_ORIGINS`, `QNT_API_URL`) before running the command. This is the fastest way to make the app reachable to other people on your network (or over the internet if the host has a public IP/reverse proxy).

### Option B — Azure Container Apps (multi-user, managed)

Ready-to-run Infrastructure-as-Code lives in [`infra/`](./infra):

- [`infra/main.bicep`](./infra/main.bicep) — Container Apps Environment (Log Analytics-backed), Azure Container Registry, a user-assigned managed identity with `AcrPull`, a Storage Account + Azure Files share mounted into the backend for persistent model storage, and two Container Apps (backend + frontend).
- [`infra/main.parameters.json`](./infra/main.parameters.json) — default parameters (region `eastus2`; edit as needed).
- [`infra/deploy.ps1`](./infra/deploy.ps1) — end-to-end deployment script: creates the resource group, deploys the Bicep template, builds both Docker images **in the cloud** via `az acr build` (no local Docker needed), then updates the Container Apps to use the real images and wires the frontend/backend URLs together.

To deploy, from a machine with the [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli) installed and logged in (`az login`):

```powershell
cd infra
./deploy.ps1 -SubscriptionId <your-subscription-id>
```

Notes and assumptions made for this template (adjust before production use):

- Targets a single shared backend replica (`maxReplicas: 1`) because model/session state is currently kept in the backend process's memory, not a shared store — scaling out would split state across replicas unpredictably.
- No GPU compute is provisioned; Container Apps here run on CPU. For GPU-accelerated inference you would need a different compute target (e.g. an Azure VM with a GPU SKU, or AKS with GPU node pools) — not covered by this template.
- `eastus2` is a reasonable default region, but availability/quota vary per subscription. If a resource fails to deploy due to region/SKU capacity, run `az account list-locations -o table` and re-deploy with a different `location`/`storageLocation` parameter.
- CORS starts wide open (`*`) so the first deploy succeeds before the frontend's URL is known; `deploy.ps1` narrows it to the real frontend URL automatically at the end.
- No secrets/API keys are used by this app today, so no Key Vault is provisioned. Add one if you later introduce credentials that must be stored securely.

## Known limitations

- QuantLab runs inference/evaluation only. It does not train or quantize models.
- Generic ONNX and TFLite model graphs are inspectable/loadable, but text generation requires architecture-specific tokenizer/generation logic not defined by those file formats. The current adapters do not pretend otherwise.
- Standalone PyTorch checkpoints are inspected in safe weights-only mode; no arbitrary pickle-based model execution is used.
- Only local causal language models through GGUF and Transformers have generic text-generation support in this version. Model-specific chat templates are not automatically inferred for all architectures.
- Exact Match and Token F1 depend on dataset wording and are lexical text-overlap measures, not semantic accuracy or benchmark accuracy from external leaderboards. Lexical output similarity does not imply semantic equivalence.
- RAM load increase uses process RSS and may include other allocations; GPU memory is not attributed per model. Peak RSS/first-token latency may be unavailable by OS/runtime.
- Streaming cancellation is effective when the runtime stream can be closed (GGUF); a non-streaming generation call may continue within its runtime after a client disconnect.
- This local demo keeps imported model/evaluation records in memory; restarting the backend clears the registry and saved evaluation IDs. Model files and built-in dataset remain on disk.
- There is no built-in upload of multi-gigabyte model weights. Place model files inside the allowed local model directory before importing.
