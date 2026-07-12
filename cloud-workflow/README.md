# Dynamic AMD Cloud Training Workflow

An interactive ML training pipeline that generates training/evaluation scripts via LLM, provisions AMD GPU Droplets on AMD DevCloud, runs the workload natively on the host (not in Docker), and downloads the results — all with minimal manual setup.

The workflow is layered: the LLM generates only a structured job plan and `script.py`. All infrastructure (VM provisioning, SSH, Python discovery, runtime stages) is handled by the deterministic Python orchestrator.

> **Current architecture note**: The workload now runs **natively on the host** (the AMD DevCloud Quick Start image has PyTorch/ROCm pre-installed), not inside Docker containers. Docker-based documentation below is preserved for legacy context — the actual execution bypasses Docker entirely and uses direct `python3` execution.

---

## Directory Structure

```
cloud-workflow/
├── .env.example                  # Template for environment variables
├── .gitignore                    # Ignores .env, __pycache__, outputs/, runtime_workspace/
├── README.md                     # This file
├── requirements.txt              # Python dependencies for the orchestrator
├── run.py                        # Main entry point — interactive flow
│
├── configs/
│   └── job_spec.example.json     # Example job spec for non-interactive flow
│
├── dynamic_cloud/                # Core package (orchestration logic)
│   ├── __init__.py
│   ├── amd_droplet.py            # AMD Droplet lifecycle management (DevCloud API)
│   ├── config.py                 # Environment loading, AmdSettings/LlmSettings dataclasses
│   ├── dataset_config.py         # Canonical dataset source normalization + HF ID resolution
│   ├── docker_runner.py          # RemoteHostRunner: SSH/SFTP operations on the VM
│   ├── executor.py               # Shared AMD execution + final report synthesis
│   ├── llm_client.py             # LLM JSON-call helpers and robust JSON response parser
│   ├── llm_generator.py          # Non-interactive LLM payload generation (used by runner.py)
│   ├── payload_quality.py        # Static analyzer + LLM review/fix prompts
│   ├── runner.py                 # Non-interactive CLI entry point (--spec, --generate, --execute)
│   ├── runtime_layers.py         # Deterministic runtime layer: image catalog, metadata, helpers
│   ├── vm_options.py             # AMD GPU catalog, pricing hints, image selection menu
│   └── workspace.py              # Local workspace creation & file management
│
├── model/                        # LLM model configuration
│   ├── __init__.py               # Lazily exports model*, make_model
│   └── model.py                  # CentralModel wrapper over the OpenAI client
│
├── runtime_workspace/            # Generated per-job working directory
│   ├── .gitkeep
│   ├── <job_id>/                 # Created at runtime (e.g. alexnet-mnist-tf-gpu)
│   │   ├── job_spec.json         # Full job specification (with amd_vm_size, amd_gpu_image)
│   │   └── payload/              # Files uploaded to the AMD Droplet
│   │       ├── metadata.json     # Runtime config: image, accelerator, packages, artifact contract
│   │       ├── config.json       # Structured job config for generated/script.py
│   │       ├── payload.json      # Dataset info (legacy name, kept for compatibility)
│   │       ├── datasets.json     # Copy of dataset config
│   │       ├── dataset_metadata.json  # Inspected dataset tree, layout, samples (when available)
│   │       ├── dataset_info.json      # Written at runtime by dataset_manager.py
│   │       ├── outputs.py             # Standardized output helper (6 functions)
│   │       ├── runtime_bootstrap.py   # Thin orchestrator for runtime stages
│   │       ├── runtime_installer.py   # Install packages and prepare/reuse dataset
│   │       ├── runtime_runner.py      # Execute generated/script.py
│   │       ├── runtime_validator.py   # Validate expected output artifacts
│   │       ├── runtime_collector.py   # Write run_manifest.json
│   │       ├── dataset_manager.py     # Deterministic dataset download/preparation
│   │       ├── dataset_inspector.py   # Deterministic dataset inspection (external datasets)
│   │       ├── data/                  # Local datasets copied here (optional)
│   │       └── generated/
│   │           ├── script.py     # The actual training/evaluation script (LLM-generated)
│   │           └── outputs.py    # Deprecated; outputs.py lives at payload root now
│   └── ... (one folder per job)
│
├── outputs/                      # Results downloaded from the AMD Droplet after run
│   ├── .gitkeep
│   ├── <job_id>/                 # One folder per completed job
│   │   ├── metrics.json          # Evaluation metrics (accuracy, loss, etc.)
│   │   ├── training_logs.json    # Per-epoch training history
│   │   ├── environment.json      # Python version, platform, package versions
│   │   ├── model.keras / model.pth / model.joblib  # Trained model weights
│   │   ├── logs/                 # Per-run log output
│   │   ├── plots/                # Saved training visualizations
│   │   └── run_manifest.json     # Artifact manifest written by runtime_collector.py
│
└── cloudENV/                     # Python virtual environment (local)
```

At the repository root, `../ml-images/` contains CPU base-image Dockerfiles and a build script:

```
ml-images/
├── build-cpu-images.sh
├── pytorch-cpu/Dockerfile
└── tensorflow-cpu/Dockerfile
```

At the project root (`final_amd/`), completed outputs and runtime workspaces are shifted from `cloud-workflow/outputs/` and `cloud-workflow/runtime_workspace/` to project-level directories:

```
final_amd/
├── outputs/<job_id>/             # Training artifacts (metrics, logs, model)
├── generated_files/<job_id>/     # Generated payload, script, metadata
└── outputs/final-report.md       # Synthesized report (research + experiment)
```

---

## Interactive Workflow (`run.py`)

Running `python run.py` starts the interactive flow:

### Step 1: User Requirement
```
What do you want to train, test, or run?
> train alexnet on mnist with tensorflow gpu
```

### Step 2: Optional — User Script Upload
If the user has a reference `.py` script with custom architecture or requirements, they can place it in `user_resources/user_script/` and the workflow will incorporate it (its content is injected into LLM prompts).

### Step 3: Framework Selection
The flow automatically selects **PyTorch** (better optimized on AMD GPU with the prebuilt Quick Start image). The user simply presses Enter to confirm.

### Step 4: Accelerator Selection
Automatically selects **GPU (AMD MI300X)**. The user presses Enter to confirm.

### Step 5: Dataset Source Selection
The user picks from four options:

1. **No external dataset** / framework built-in
2. **Hugging Face dataset** — provide a dataset ID (e.g. `ylecun/mnist`)
3. **GitHub repository** — provide a GitHub URL
4. **Local file or folder path** — provide one or more local paths

For local datasets, files are copied to a permanent staging directory at `user_resources/uploading_data/` and inspected entirely **locally** — no VM needed.

### Step 6: LLM Generates Clarifying Questions
The user's requirement, selected framework/accelerator, dataset source, and optional user script are sent to the LLM with the `QUESTION_PROMPT` system prompt. The LLM returns 3–5 JSON questions for missing details (e.g., epochs, batch size, learning rate, optimizer, architecture variant). It is explicitly told not to ask about framework, accelerator, dataset source, or VM size — those are handled separately by the orchestrator.

### Step 7: User Answers Questions
Each question is displayed with an optional default value:
```
1. How many epochs? [5]
> 5
2. What batch size? [64]
> 64
3. Any specific Python packages besides tensorflow?
> matplotlib numpy
```

### Step 8: VM Selection
The user chooses from a predefined catalog of AMD GPU options:

1. `gpu-mi300x1-192gb-devcloud` — 1× AMD MI300X, 192 GB VRAM (default)
2. `gpu-mi300x8-1536gb-devcloud` — 8× AMD MI300X, 1.5 TB VRAM

The user may also select the Droplet **image** (OS/framework preload):
1. **PyTorch** (v2.10.0 — recommended)
2. Others (ROCm base, vLLM, SGLang, JAX, Ubuntu)

### Step 9: Remote Dataset Inspection (External Datasets)
For external datasets (Hugging Face, GitHub), the workflow starts the selected VM before code generation and runs a deterministic inspect-only payload:

```
User answers + dataset source + VM
  → create Droplet (cloud-init installs python3-pip)
  → wait for SSH readiness, vendor provisioning, and stable post-boot SSH
  → silence vendor shell banner
  → upload payload via tar-over-SSH pipe
  → probe remote Python interpreter (system python3 → conda envs → brute force)
  → download/prepare dataset with dataset_manager.py (source-hash cached on VM)
  → inspect actual files with dataset_inspector.py
  → persist dataset cache on VM (under /home/<admin>/dynamic_jobs/datasets/)
  → write dataset_metadata.json
  → download dataset_metadata.json locally
```

The inspection Droplet is **reused** for the immediate training run so the prepared dataset cache stays warm.

### Step 10: LLM Generates, Reviews, and Repairs the Job Plan and Script
The user's context, Q&A session, selected VM, dataset source, and inspected `dataset_metadata.json` are sent to the LLM with the `GENERATION_PROMPT` system prompt. The LLM returns a JSON payload containing:

- **`job_spec`**: structured job metadata (job_id, objective, task_type, dataset info, runtime config, expected outputs)
- **`files`**: one generated file:
  - **`generated/script.py`**: The actual ML training/evaluation code
- **`config`**: additional configuration (includes `dataset_metadata` when available)
- **`summary`**: task description, training plan, estimated runtime

The LLM is instructed to:
- Use the **outputs.py** helper functions (save_metrics, log_epoch, save_training_history, save_plot, save_model, save_environment)
- Never hardcode `/outputs/` paths — resolve via `DYNAMIC_CLOUD_OUTPUTS_DIR` env var
- Avoid deprecated PyTorch arguments (e.g., `verbose=True` removed in v2.4+)
- Import `ImageFolder` from `torchvision.datasets`, not `torch.utils.data`
- Not generate Dockerfiles or requirements.txt
- Not include dataset download logic in `script.py`
- Not list torch/tensorflow in extra_packages (they are pre-installed)
- Escaping JSON string values correctly (no raw newlines inside JSON strings)

#### Quality Gate

Before the payload reaches the AMD Droplet, it passes through a multi-stage quality gate:

```
Generate → Static analyzer → LLM review → Fix if needed → Validate → Execute
```

**Static analyzer** checks:
- JSON shape and syntax
- Forbidden Dockerfile/requirements.txt in layered workflow
- Framework/accelerator drift between context and generated spec
- Requested epoch count is present and used in training
- Hallucinated output helper imports (auto-fixed)
- Dataset download patterns in script (forbidden)
- TensorFlow data antipatterns (e.g., `output_types` vs `output_signature`)
- Training scripts call `save_model()` (required)
- Common hallucinated imports (e.g., `torch.utils.data.ImageFolder`)

**LLM reviewer** compares the generated code against the original requirement.

**Repair loop** (up to 2 rounds): if issues are found, the LLM generates a corrected full payload.

### Step 11: Workspace Created Locally

`prepare_workspace()` creates the job folder structure. Then `write_layered_payload()` writes all runtime files:

```
runtime_workspace/<job_id>/
├── job_spec.json           # Full spec with VM info, dataset, runtime config
└── payload/
    ├── metadata.json        # Runtime metadata (framework, accelerator, image, packages, artifact contract)
    ├── config.json          # Structured runtime config (including dataset_metadata)
    ├── payload.json         # Dataset info (legacy name)
    ├── datasets.json        # Copy of dataset config
    ├── dataset_metadata.json # Inspected real dataset tree & layout (when available)
    ├── outputs.py           # Standardized output helper (6 functions)
    ├── runtime_bootstrap.py # Thin orchestrator for runtime stages
    ├── runtime_installer.py # Install packages and prepare/reuse dataset
    ├── runtime_runner.py    # Execute generated/script.py
    ├── runtime_validator.py # Validate expected outputs
    ├── runtime_collector.py # Write run_manifest.json
    ├── dataset_manager.py   # Deterministic dataset download/preparation
    ├── dataset_inspector.py # Deterministic dataset inspection
    ├── data/                # Local dataset files copied here
    └── generated/
        ├── script.py        # ML training/evaluation code (LLM-generated)
        └── outputs.py       # Deprecated; use payload root outputs.py
```

Dataset paths listed in `job_spec["dataset"]["local_paths"]` are copied into `payload/data/` before upload.

### Step 12: Summary & Confirmation

A summary is printed with the job ID, task description, dataset plan, selected VM, runtime image, packages, estimated runtime and cost, expected outputs, and notes.

```
Start AMD GPU Droplet and run training now? [Y/n]
```

### Step 13: AMD Execution

Execution is handled by `execute_workspace_on_amd()` in `dynamic_cloud/executor.py`:

1. **`AmdDropletManager.create_droplet()` or `adopt()`** (`dynamic_cloud/amd_droplet.py`):
   - Creates a Droplet via the AMD DevCloud API with the specified GPU plan and image
   - Supports multi-size fallback across multiple regions
   - Automatically registers/finds SSH keys
   - Assigns the Droplet to a DigitalOcean project (best-effort)
   - Alternatively, adopts an existing inspection Droplet or a UI-created Droplet

2. **`AmdDropletManager.wait_until_active()`**: Polls the API until the status is "active", then waits for SSH (TCP port 22 + key-based `ssh` login).

3. **`RemoteHostRunner.upload_workspace()`** (`dynamic_cloud/docker_runner.py`):
   - Waits for authenticated SSH and vendor provisioning (`"Please wait"` banner gone)
   - Silences the vendor shell banner to avoid SSH output pollution
   - Streams a tar archive of the payload directory over SSH (tar pipe — avoids SCP fragility)

4. **`RemoteHostRunner.execute_bootstrap()`**:
   - **Discovers the remote Python interpreter** with PyTorch available:
     1. System `python3` (fast path)
     2. Conda environments (`conda env list` → probe each)
     3. Brute-force `python3` binaries under `/opt`, `/usr/local`, `/root`
     4. Locate `torch/__init__.py` and derive the Python interpreter path
   - If no Python with torch is found, **installs ROCm PyTorch** into a package overlay at `/opt/dynamic-cloud/python-packages/` from `https://download.pytorch.org/whl/rocm7.2`
   - Executes `python3 -u runtime_bootstrap.py` on the host

5. **Staged runtime execution** (`runtime_bootstrap.py`):
   - `runtime_installer.py` — installs missing packages (into overlay), prepares/reuses the dataset
   - `runtime_runner.py` — executes `generated/script.py` in-process with `exec()`
   - `runtime_validator.py` — checks required output artifacts from the contract in `metadata.json`
   - `runtime_collector.py` — writes `run_manifest.json` with all produced artifacts

6. **`RemoteHostRunner.download_outputs()`**: Downloads the `outputs/` directory via a tar-over-SSH pipe (fresh connection, no multiplex socket for reliability).

7. **Cleanup**: Destroys the Droplet to stop billing. GPU Droplets bill per-second with a 5-minute minimum — only destroying stops billing.

8. **Post-execution**: Shifts outputs and runtime workspace to project-level directories (`final_amd/outputs/<job_id>/`, `final_amd/generated_files/<job_id>/`).

### Step 14: Final Report Generation

If the research module (`src.ai.providers`) is available, a comprehensive `final-report.md` is synthesized from:
- Any existing web research report (`outputs/report.md`)
- Cloud experiment outputs (metrics, training logs, environment, runtime log)
- Experiment plots (loss/accuracy curves)
- Research report images

The report is saved to `final_amd/outputs/final-report.md`.

---

## Non-Interactive Flow (`runner.py`)

`python -m dynamic_cloud.runner` provides a scriptable CLI interface:

```bash
# Prepare workspace from a job spec JSON
python -m dynamic_cloud.runner --spec configs/job_spec.example.json --generate

# Generate and validate
python -m dynamic_cloud.runner --spec configs/job_spec.example.json --generate --validate

# Full run (generate + validate + create AMD Droplet + run + cleanup)
python -m dynamic_cloud.runner --spec configs/job_spec.example.json --generate --execute

# Keep the Droplet after run for debugging
python -m dynamic_cloud.runner --spec configs/job_spec.example.json --generate --execute --keep-vm
```

Flags:
| Flag | Purpose |
|------|---------|
| `--spec PATH` | **Required.** Path to job spec JSON |
| `--generate` | LLM generates structured payload and script.py |
| `--validate` | Check generated payload files |
| `--execute` | Provision AMD Droplet and run the job |
| `--reset` | Recreate the local workspace |
| `--keep-vm` | Skip Droplet cleanup after run (billing continues) |
| `--vm-name NAME` | Override the AMD Droplet name |

---

## Prepared Job Commands (`run.py`)

```bash
# Run an already-prepared job (skips generation, Q&A, VM selection)
python run.py --execute-prepared runtime_workspace/alexnet-mnist-tf-gpu

# Override VM size when using --execute-prepared
python run.py --execute-prepared runtime_workspace/alexnet-mnist-tf-gpu --vm-size gpu-mi300x8-1536gb-devcloud

# Reuse a GPU Droplet created from the AMD DevCloud UI
python run.py --amd-droplet-id 123456789 --amd-droplet-ip 203.0.113.10

# Run a prepared job on an existing UI-created Droplet
python run.py --execute-prepared runtime_workspace/alexnet-mnist-tf-gpu --amd-droplet-id 123456789

# Clean up AMD GPU Droplet for a prepared job
python run.py --cleanup-prepared runtime_workspace/alexnet-mnist-tf-gpu
```

---

## Execution Architecture

### Native Host Execution (Not Docker)

The AMD DevCloud Quick Start images already have PyTorch 2.10.0 + ROCm 7.2.4 pre-installed. Rather than layering Docker on top, the runner:

1. **Discovers the right Python interpreter** on the remote host — probes system python3, conda environments, and brute-force search
2. **Installs a package overlay** at `/opt/dynamic-cloud/python-packages/` for any extra packages (never modifies the base system site-packages)
3. If PyTorch is absent (ROCm base image), **installs ROCm PyTorch** from the AMD PyTorch wheel index
4. **Executes the generated script in-process** via `exec()`, inheriting all env vars, sys.path entries, and installed modules

### Python Discovery Strategy

The `_discover_remote_python()` method probes in this order:

1. **System `python3`** — fast path for images that install torch to system Python
2. **Conda environments** — AMD DevCloud PyTorch Quick Start images often install inside conda (non-interactive SSH doesn't source .bashrc, so conda isn't activated by default)
3. **Brute-force** — finds any `python3` under `/opt`, `/usr/local`, `/root` that can import torch
4. **Torch module location** — finds `torch/__init__.py` and derives the Python interpreter path

### ROCm PyTorch Fallback

If no Python with `torch` is found:

```
pip install torch torchvision torchaudio \
  --target /opt/dynamic-cloud/python-packages \
  --index-url https://download.pytorch.org/whl/rocm7.2
```

### Output Download Reliability

The download uses a fresh SSH connection (no ControlMaster multiplex socket) to avoid stale connection issues. The remote `tar` is wrapped in `timeout` so a stuck remote process cannot hang the channel. The local `tar` extraction runs concurrently with the SSH pipe.

---

## Dataset Management

### Dataset Sources

| Type | Identifier | Container Path |
|------|-----------|----------------|
| `none` | — | No dataset |
| `huggingface` | `ylecun/mnist` (HF dataset ID) | `/workspace/prepared_datasets/<hash>/` |
| `github` | `https://github.com/user/dataset` | `/workspace/prepared_datasets/<hash>/` |
| `local` | Absolute file/folder paths | `/workspace/data/` |

### Hugging Face ID Resolution

Bare dataset names (e.g., `beans`) are auto-resolved to namespace/name format (e.g., `AI-Lab-Makerere/beans`) via the Hugging Face Hub API. This avoids `HfUriError` on newer `huggingface_hub` versions.

### VM-Level Dataset Cache

Downloaded external datasets are cached on the VM at `/home/<admin>/dynamic_jobs/datasets/` using a SHA-256 hash of the normalized dataset source. When the inspection Droplet is reused for training, the cache is preserved — no re-download needed.

### Local Dataset Staging

Local datasets are staged to a persistent directory at `user_resources/uploading_data/` and inspected **entirely locally** — no VM needed. The inspection produces `dataset_metadata.json` with the same format as remote inspection.

### Local Dataset Inspection

When the user selects "Local file or folder path":

1. Files are copied to `user_resources/uploading_data/` (persistent across runs)
2. The `_inspect_local_dataset()` method scans the directory locally:
   - Counts files, directories, and file extensions
   - Detects layout: split directories (train/val/test), class directories, or tabular/annotation files
   - Samples text files (CSV, JSON, YAML, etc.)
3. Returns `dataset_metadata.json` in the same format as remote inspection

### Remote Dataset Inspection

For Hugging Face and GitHub datasets, the workflow:

1. Creates a Droplet (or reuses an existing one)
2. Uploads a minimal inspection payload
3. Executes `dataset_inspector.py` which:
   - Downloads/prepares the dataset using `dataset_manager.py`
   - Walks the directory tree (bounded to depth 5, 80 entries per level)
   - Counts files by extension
   - Detects split/class-folder layout
   - Inspects Hugging Face datasets (splits, features, types)
   - Samples CSV/JSON files
4. Downloads `dataset_metadata.json` via SSH+Python (immune to shell banner issues)

---

## Quality Gate

The quality gate (`dynamic_cloud/payload_quality.py`) is a multi-stage guard between LLM generation and AMD execution:

### Static Analyzer Checks

| Check | What it verifies |
|-------|-----------------|
| JSON shape | `job_spec` and `files` objects exist |
| Syntax | `generated/script.py` parses as valid Python AST |
| Placeholder text | No "placeholder" or "no runtime script was generated" |
| Forbidden files | No Dockerfile or requirements.txt in layered workflow |
| Framework/accelerator | Matches the user's selections |
| Epoch count | User-requested epoch count is used in training |
| Framework imports | TensorFlow scripts import tensorflow/keras; PyTorch scripts import torch |
| Output helper imports | Only the 6 valid functions (save_metrics, log_epoch, etc.) |
| `save_model()` | Training jobs call save_model() — required for model.pth |
| Dataset download | No `datasets.load_dataset()`, `git clone`, etc. in script.py |
| TensorFlow antipatterns | `output_types` vs `output_signature` in image generators; `np.array()` inside `.map()` |
| Hallucinated imports | `torch.utils.data.ImageFolder` → should be `torchvision.datasets.ImageFolder` |
| Output artifacts | Script writes to /outputs directory or uses outputs helper |

### LLM Review

A second LLM call (using `model1`) reviews the payload against the user requirements, checking:
- The script satisfies the user's task description
- Dataset handling matches the inspected metadata
- Output helper usage is correct
- No download logic in training scripts

### Repair Loop

If the static analyzer or LLM reviewer finds issues (up to 2 rounds):
1. The repair LLM is called with context + the issues
2. It returns a full corrected payload
3. The static analyzer and LLM reviewer run again on the corrected payload

---

## Runtime Stages

`runtime_bootstrap.py` orchestrates these stages in order:

### 1. `runtime_installer.py`

- Reads `metadata.json` for extra_packages
- Filters out pre-installed DL frameworks (torch, tensorflow, etc.) and blocked CUDA packages
- Checks if each package is already importable
- Installs missing packages into `/opt/dynamic-cloud/python-packages/` overlay
- Prepares the dataset via `dataset_manager.py`

### 2. `runtime_runner.py`

- Reads the entrypoint from `metadata.json`
- Sets environment variables (`DATASET_PATH`, `DATASET_INFO_PATH`, `DATASET_METADATA_PATH`)
- Adds the workspace to `sys.path`
- Executes `generated/script.py` **in-process** via `exec()` so it inherits all env vars and installed modules

### 3. `runtime_validator.py`

- Checks required output files and directories from the artifact contract in `metadata.json`
- Reports OK/MISS/WARN for each artifact
- Required by default: `environment.json` (always)
- Required for training jobs: `metrics.json`, `training_logs.json`
- Required for eval jobs: `metrics.json`
- Optional: `logs/`, `plots/`
- Custom expected outputs from `job_spec.expected_outputs`
- Writes `logs/output_report.txt`
- **Exits with error** if any required artifacts are missing

### 4. `runtime_collector.py`

- Walks all files in the `OUTPUTS` directory
- Writes `run_manifest.json` with paths and sizes of all artifacts

---

## Standardized Output Helper (`outputs.py`)

Every payload includes `outputs.py` (at the payload root, NOT under `generated/` — placed there so `from outputs import save_metrics` resolves before the `outputs/` data directory). It provides 6 functions:

| Function | Purpose | Writes to |
|----------|---------|-----------|
| `save_metrics(metrics: dict)` | Save evaluation metrics (accuracy, loss, etc.) | `OUTPUTS/metrics.json` |
| `log_epoch(epoch, metrics)` | Append per-epoch training metrics | `OUTPUTS/training_logs.json` |
| `save_training_history(history)` | Save full Keras/PyTorch training history | `OUTPUTS/training_logs.json` |
| `save_plot(figure, name)` | Save a matplotlib figure | `OUTPUTS/plots/<name>` |
| `save_model(model, path, framework)` | Save trained model | `OUTPUTS/<path>.<ext>` |
| `save_environment(extra)` | Snapshot Python version, platform, and packages | `OUTPUTS/environment.json` |

All paths resolve via `DYNAMIC_CLOUD_OUTPUTS_DIR` environment variable (never hardcoded `/outputs/`). `save_model()` returns the OUTPUTS directory (not the file path) so `os.path.join(result, "model.pth")` resolves correctly.

---

## Image Catalog

The `IMAGE_CATALOG` in `runtime_layers.py` determines which runtime image reference is used for the artifact contract:

| Framework | Accelerator | Image |
|-----------|------------|-------|
| PyTorch | CPU | `abhaya123/pytorch-cpu:latest` |
| TensorFlow | CPU | `abhaya123/tensorflow-cpu:latest` |
| PyTorch | GPU | `rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_2.10.0` |
| TensorFlow | GPU | `rocm/tensorflow:rocm7.2.4` |

Override each via environment variables: `PYTORCH_CPU_IMAGE`, `TENSORFLOW_CPU_IMAGE`, `PYTORCH_GPU_IMAGE`, `TENSORFLOW_GPU_IMAGE`.

---

## AMD Droplet Image Selection

The Droplet image (the OS/software preload on the VM, vs. the runtime image reference in metadata) can be selected interactively. The available images and their topology-aware slugs are defined in `vm_options.py` and `AMD_GPU_IMAGES`:

| Image | Slug (1×) | Slug (8×) |
|-------|-----------|-----------|
| PyTorch v2.10.0 (recommended) | `amddevelopercloud-pytorch2100rocm724` | (same) |
| ROCm 7.2.4 (base) | `rocm-7-2-4` | (same) |
| vLLM v0.23.0 | `amddeveloperclou-vllm0230rocm724` | (same) |
| SGLang v0.5.14 | `amddeveloperclou-sglang0514rocm72` | `sglang-0-5-14-rocm-7-2-4` |
| JAX v0.8.2 | `amddeveloperclou-jax082rocm724` | `jax-0-8-2-rocm-7-2-4` |
| Ubuntu 26.04 x64 | `ubuntu-26-04-x64` | (same) |

Override via environment variables: `AMD_GPU_IMAGE`, `AMD_PYTORCH_GPU_IMAGE`, `AMD_TENSORFLOW_GPU_IMAGE`.

---

## End-to-End Data Flow

```
User input ("train alexnet on mnist with tensorflow gpu")
    │
    ├── Optional: User has their own script? → read from user_resources/user_script/
    │
    ▼
Framework: PyTorch (auto-selected)
Accelerator: GPU (auto-selected)
Dataset source: None / HuggingFace / GitHub / Local
    │
    ▼
LLM (QUESTION_PROMPT) ───→ 3-5 clarifying questions (epochs, batch size, LR, etc.)
    │
    ▼
User answers questions
    │
    ▼
User selects AMD GPU plan from catalog (and optionally the Droplet image)
    │
    ├──[external dataset?]──── yes ──────────────────────────────────────────┐
    │                                                                         │
    │  Remote Dataset Inspection:                                             │
    │    → AmdDropletManager.create_droplet() or adopt()                     │
    │    → wait_until_active() + get_public_ip()                             │
    │    → RemoteHostRunner.upload_workspace() (tar pipe over SSH)            │
    │    → run_dataset_inspection():                                          │
    │        probe remote Python → install dataset packages                   │
    │        → dataset_manager.py (prepare/reuse dataset)                    │
    │        → dataset_inspector.py (walk tree, detect layout, sample files) │
    │        → persist dataset cache on VM                                   │
    │        → download_dataset_metadata() (SSH+Python, immune to banners)    │
    │    → Droplet stays alive for training run                              │
    │    ├──[local dataset?]──── yes ───────────────────────────────────────┐ │
    │    │                                                                   │ │
    │    │  Local Dataset Inspection (no VM needed):                        │ │
    │    │    → Copy files to user_resources/uploading_data/                 │ │
    │    │    → _inspect_local_dataset(): walk tree, detect layout           │ │
    │    │    → Write dataset_metadata.json locally                         │ │
    │    └───────────────────────────────────────────────────────────────────┘ │
    └───────────────────────────────────────────────────────────────────────────┘
    │
    ▼
LLM (GENERATION_PROMPT) ───→ JSON with job_spec + generated/script.py
    │
    ▼
Quality Gate (payload_quality.py + LLM):
    Static analyzer (AST checks):
      framework imports, epoch count, /outputs writes,
      forbidden Dockerfile/requirements, dataset download patterns,
      TF data antipatterns, hallucinated imports, save_model() call
    ───→ LLM reviewer (REVIEW_PROMPT) ───→ [issues?] ───→ LLM fixer (FIX_PROMPT, up to 2 rounds)
    ───→ merged static + review report → pass or reject
    │
    ▼
write_layered_payload() (runtime_layers.py):
    → _validate_outputs_imports() (auto-fix hallucinated output helper imports)
    → build_runtime_metadata() → imports_from_python() (AST scan) → extra_packages
    → select_image() → artifact contract → required/optional outputs
    → write: metadata.json, config.json, payload.json, datasets.json
    → write: outputs.py, runtime_bootstrap.py → runtime_installer.py → runtime_runner.py
              runtime_validator.py, runtime_collector.py
    → write: dataset_manager.py, dataset_inspector.py
    → write: generated/script.py, generated/outputs.py (deprecated)
    → clean stale Dockerfile/requirements.txt
    │
    ▼
Local workspace ready at runtime_workspace/<job_id>/payload/
    │
    ▼
AMD Execution (executor.py → execute_workspace_on_amd):
    │
    ├── AmdDropletManager.create_droplet() or adopt():
    │    → creates droplet using DO-style API, or reuses the dataset inspection Droplet
    │
    ├── wait_until_active(): poll active state + wait for SSH
    │
    ├── RemoteHostRunner.upload_workspace():
    │    → wait for SSH + vendor provisioning → silence shell banner
    │    → tar payload/ → SSH pipe → extract on VM
    │
    ├── execute_bootstrap():
    │    → _discover_remote_python() (system → conda → brute force → torch module)
    │    → [if no torch found: install ROCm PyTorch overlay]
    │    → _run_host_stage():
    │        mkdir outputs, ln -sfn dataset cache
    │        [if inspection: install dataset packages]
    │        export DYNAMIC_CLOUD_* env vars
    │        python3 -u runtime_bootstrap.py (pipe stdout to outputs/logs/runtime.log)
    │    → runtime_bootstrap.py stages:
    │        1. runtime_installer.py → pip install missing packages → prepare_dataset
    │        2. runtime_runner.py → exec(generated/script.py) in-process
    │        3. runtime_validator.py → check artifact contract
    │        4. runtime_collector.py → write run_manifest.json
    │
    ├── RemoteHostRunner.download_outputs():
    │    → fresh SSH connection → timeout tar pipe → local extract
    │
    ├── [optional] _generate_final_report():
    │    → synthesize research + experiment → final-report.md
    │
    └── AmdDropletManager.destroy():
         └── deletes Droplet to stop billing
```

---

## What's Dynamic vs Static

| Aspect | Dynamic | Static |
|--------|---------|--------|
| Training script | Generated by LLM per request | — |
| Framework/accelerator plan | User-selected and recorded in job spec | — |
| Extra packages | LLM-declared and import-scanned | Installed only when missing, into overlay |
| Dataset | User-specified or LLM-chosen | Downloaded/reused through VM-level source-hash cache |
| Output artifacts | Defined by LLM in artifact contract | — |
| Droplet image | User-chosen (PyTorch, ROCm, vLLM, etc.) | Topology-aware slug resolution |
| Output helper | — | `outputs.py` provides 6 standardized functions |
| Runtime stages | — | Same installer → runner → validator → collector every time |
| AMI base dependencies | — | Pre-installed in AMD DevCloud Quick Start images |
| AMD resource creation | — | Same API call every time |
| VM lifecycle | — | Create/adopt → Wait → Use → Destroy |
| SSH/SCP transport | — | Same every time (tar pipe) |
| Output download | — | Same SSH pipe every time |

---

## Environment Variables

### AMD DevCloud / DigitalOcean
| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `AMD_API_KEY` | Yes* | — | AMD DevCloud API key. Also accepted: `DIGITALOCEAN_ACCESS_TOKEN`, `DO_API_TOKEN`, `AMD_TOKEN` |
| `AMD_API_BASE_URL` | No | `https://api.devcloud.amd.com/v2` | AMD DevCloud API endpoint |
| `AMD_PROJECT_ID` | No | — | Project UUID; assignment is best-effort |
| `AMD_REGION` | No | `atl1` | AMD DevCloud region |
| `AMD_REGIONS` | No | `atl1` | Comma-separated fallback region list |
| `AMD_DEFAULT_VM_SIZE` | No | `gpu-mi300x1-192gb-devcloud` | Default GPU plan |
| `AMD_VM_SIZES` | No | `AMD_DEFAULT_VM_SIZE` | Comma-separated fallback creation list |
| `AMD_VM_NAME` | No | `dynamic-ml-droplet` | Base instance name |
| `AMD_GPU_IMAGE` | No | `rocm-7-2-4` | Droplet image slug override (GPU) |
| `AMD_PYTORCH_GPU_IMAGE` | No | `amddevelopercloud-pytorch2100rocm724` | PyTorch GPU Droplet image |
| `AMD_TENSORFLOW_GPU_IMAGE` | No | `amddeveloperclou-rocm724` | TensorFlow GPU Droplet image |
| `AMD_ADMIN_USER` | No | `root` | VM admin username |
| `AMD_SSH_KEY_NAME` | No | — | Registered SSH key name/ID to look up |
| `AMD_SSH_KEYS` | No | — | Comma-separated existing SSH key IDs/names |
| `AMD_REGISTER_SSH_KEY` | No | `0` | Set to `1` to auto-register public key |
| `AMD_SSH_PUBLIC_KEY_PATH` | No | `~/.ssh/id_rsa.pub` | Local SSH public key path |
| `AMD_SSH_PRIVATE_KEY_PATH` | No | `~/.ssh/id_rsa` | Local SSH private key path |
| `AMD_VPC_UUID` | No | `8244dc95-...` | VPC UUID for create payload |
| `AMD_TAGS` | No | — | Comma-separated Droplet tags |
| `AMD_EXISTING_DROPLET_ID` | No | — | Reuse an existing Droplet ID instead of creating |
| `AMD_EXISTING_DROPLET_IP` | No | — | Public IPv4 for `AMD_EXISTING_DROPLET_ID` |
| `AMD_DROPLET_WAIT_TIMEOUT_SEC` | No | `600` | Droplet active wait timeout (seconds) |
| `AMD_SSH_WAIT_TIMEOUT_SEC` | No | `300` | SSH readiness wait timeout (seconds) |
| `AMD_INITIALIZATION_WAIT_TIMEOUT_SEC` | No | `1200` | Cloud-init/vendor provisioning wait timeout |
| `AMD_DOCKER_BASE_IMAGE` | No | — | Legacy — overrides Docker base image |
| `AMD_DOCKER_RUNTIME_IMAGE` | No | — | Legacy — overrides Docker runtime image tag |

### LLM / Fireworks AI
| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `FIREWORKS_API_KEY` | **Yes** | — | Fireworks AI API key |
| `MODEL1` | No | `accounts/fireworks/models/minimax-m2p7` | Model for slot 1 (main generation, review, fix) |
| `MODEL2` | No | `accounts/fireworks/models/minimax-m2p7` | Model for slot 2 (not used by default) |
| `FIREWORKS_BASE_URL` | No | `https://api.fireworks.ai/inference/v1` | Fireworks API base URL |

### Runtime Overrides
| Variable | Default | Purpose |
|----------|---------|---------|
| `PYTORCH_CPU_IMAGE` | `abhaya123/pytorch-cpu:latest` | Override PyTorch CPU runtime image reference |
| `TENSORFLOW_CPU_IMAGE` | `abhaya123/tensorflow-cpu:latest` | Override TensorFlow CPU runtime image reference |
| `PYTORCH_GPU_IMAGE` | `rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_2.10.0` | Override PyTorch GPU runtime image reference |
| `TENSORFLOW_GPU_IMAGE` | `rocm/tensorflow:rocm7.2.4` | Override TensorFlow GPU runtime image reference |
| `DYNAMIC_CLOUD_DEBUG_COMMANDS` | (not set) | Set to `1` to print all SSH/SCP commands |
| `DYNAMIC_CLOUD_SSH_RETRIES` | `5` | Number of transient SSH error retries |

---

## File-by-File Explanation

### Root Files

- **`.env.example`** — Template for credentials. Copy to `.env` and fill in `AMD_API_KEY`, `FIREWORKS_API_KEY`, SSH key settings, and optionally model overrides.

- **`requirements.txt`** — Python packages needed on the orchestrator machine: `requests`, `openai`, `python-dotenv`.

- **`run.py`** — The main interactive entry point. Contains the full interactive flow: user prompt → user script check → framework/accelerator selection → dataset selection → LLM questions → user answers → VM selection → Droplet image selection → dataset inspection (local or remote) → LLM generation → quality gate → AMD execution → output download → cleanup. Supports `--execute-prepared <job_dir>`, `--vm-size <size>`, `--amd-droplet-id <id>` / `--amd-droplet-ip <ip>`, and `--cleanup-prepared <job_dir>`.

### configs/

- **`job_spec.example.json`** — A template job spec showing the expected JSON structure for the non-interactive `runner.py` flow.

### dynamic_cloud/ (Core Package)

- **`__init__.py`** — Package marker with docstring.

- **`config.py`** — Defines `AmdSettings`, `AmdDropletSession`, and `LlmSettings` dataclasses. Contains `load_environment()` (reads `.env` with protection for already-set keys), `load_amd_settings()` (builds `AmdSettings` from env vars with multi-key fallback), and `load_llm_settings()`.

- **`vm_options.py`** — AMD GPU catalog (`VM_OPTIONS`), pricing per hour (`AMD_GPU_PRICING_PER_HOUR`), available DevCloud images (`AMD_GPU_IMAGES`), interactive image selection menu (`select_amd_image()` with topology-aware slugs for 1× vs 8×), and GPU compatibility validation.

- **`dataset_config.py`** — Canonical dataset source normalization. `resolve_hf_dataset_id()` resolves bare HF names to namespace/name form via the Hugging Face Hub API (with caching). `normalize_dataset_config()` standardizes dataset configs from various input forms into a canonical `{type, id, url, local_paths, container_data_dir}` shape.

- **`workspace.py`** — Manages the local runtime workspace for each job. `JobWorkspace` dataclass holds paths. `prepare_workspace()` creates the directory structure and writes `job_spec.json`. `copy_dataset_paths()` copies local dataset files into `payload/data/`. `validate_payload()` checks all required files exist, are non-empty, compile as valid Python, and that layered payloads contain no stale Dockerfile/requirements.txt. `_refresh_layered_helpers()` re-writes runtime stage files.

- **`amd_droplet.py`** — `AmdDropletManager` handles the full Droplet lifecycle:
  - `create_droplet()` — tries each VM size across regions with fallback
  - `ensure_ssh_key()` — auto-registers SSH public keys with the DevCloud account
  - `wait_until_active()` — polls API + waits for SSH
  - `adopt()` — attaches to an existing Droplet for reuse/cleanup
  - `destroy()` — deletes the Droplet (3 retries on failure)
  - `_request()` — rate-limit-aware HTTP helper (429 retry with `Retry-After`)

- **`docker_runner.py`** — `RemoteHostRunner` manages remote operations on the VM via SSH/SFTP:
  - `upload_workspace()` — tar-over-SSH pipe (avoids SCP banner fragility)
  - `_wait_for_ssh_ready()` — probes every 10s for basic SSH (20 attempts)
  - `_wait_for_provisioning()` — waits for AMD vendor init to finish (42 attempts)
  - `_silence_shell_banner()` — neutralizes `"Please wait"` vendor MOTD
  - `_discover_remote_python()` — probes system → conda → brute force → torch module
  - `execute_bootstrap()` — runs the staged payload natively on the host
  - `download_outputs()` — tar-over-SSH pipe with fresh connection and timeout guard
  - `_run_host_stage()` — sets env vars, creates outputs/dataset cache dirs, executes command
  - Transient SSH error handling (auto-retry with exponential backoff)

- **`runtime_layers.py`** — Deterministic runtime layer generation:
  - `IMAGE_CATALOG` — framework/accelerator → image reference mapping
  - `normalize_framework()` / `normalize_accelerator()` — string normalization with aliases
  - `select_image()` — looks up image from catalog
  - `imports_from_python()` — AST-based import scanning
  - `missing_runtime_packages()` — determines packages not already available
  - `_filter_preinstalled_frameworks()` — strips CUDA/NVIDIA/AMD-core packages
  - `build_runtime_metadata()` — builds artifact contract (required/optional outputs/dirs)
  - `_validate_outputs_imports()` — AST-based fix for hallucinated output helper imports
  - `write_layered_payload()` — writes all runtime files, metadata, configs
  - `write_dataset_inspection_payload()` — writes minimal inspection payload
  - `dataset_package_install_code()` — generates the dataset package install script
  - All runtime stage sources as Python string templates:
    - `_outputs_helper_source()` — 6 standardized output functions
    - `_dataset_manager_source()` — deterministic dataset download/preparation
    - `_dataset_inspector_source()` — directory tree walking and layout detection
    - `_runtime_installer_source()` — package overlay installation
    - `_runtime_runner_source()` — in-process script execution
    - `_runtime_validator_source()` — artifact contract validation
    - `_runtime_collector_source()` — artifact manifest generation
    - `_bootstrap_source()` — stage orchestrator

- **`payload_quality.py`** — Multi-stage quality gate between generation and execution:
  - `analyze_generated_payload()` — static AST-based analysis covering all checks
  - `REVIEW_PROMPT` / `FIX_PROMPT` — LLM reviewer and repair prompt templates
  - `review_approved()` — combines static + LLM review results
  - `build_review_request()` / `build_fix_request()` — context builders
  - `_check_outputs_imports()` — validates output helper imports
  - `_check_common_import_errors()` — catches known hallucinated imports
  - `_validate_save_model_call()` — ensures training scripts persist the model
  - `_check_tf_data_antipatterns()` — catches `output_types` + `tf.image.resize` and `np.array()` in `.map()`
  - `_epoch_constant_used_for_training()` — AST-based training epoch verification
  - `_script_mentions_epoch_count()` — checks epoch constant assignments

- **`llm_client.py`** — Shared LLM JSON-call helpers:
  - `generate_json()` / `generate_json_with_model()` — send prompts and parse responses
  - `parse_json_response()` — robust JSON parser with markdown code-block stripping, balanced-bracket fallback, and control-character sanitization inside strings
  - `_sanitize_json_strings()` — fixes LLM responses with real newlines in JSON strings

- **`llm_generator.py`** — Non-interactive LLM payload generation (`SYSTEM_PROMPT`). Wraps the same quality gate logic (generate → static analyze → LLM review → fix loop) used by `run.py`.

- **`runner.py`** — Non-interactive CLI entry point (`python -m dynamic_cloud.runner`). Accepts `--spec`, `--generate`, `--execute`, `--validate`, `--reset`, `--keep-vm`, `--vm-name`. Orchestrates the same steps as `run.py` but without interactivity.

- **`executor.py`** — Shared AMD execution backend:
  - `execute_workspace_on_amd()` — orchestrates Droplet creation/adoption, upload, execution, output download, and cleanup with comprehensive error handling
  - `_shift_completed_outputs()` — moves outputs and workspace to project-level directories after completion
  - `_generate_final_report()` — synthesizes a comprehensive `final-report.md` from research + experiment results using an LLM, embedding experiment plots and research images
  - `_extract_sources_from_research()` — extracts URLs from research report sources section

### model/

- **`model.py`** — Central LLM model configuration:
  - `CentralModel` wraps an OpenAI-compatible client with `complete_json()` and `invoke()` methods
  - `model1` / `model2` — two configurable model slots (both default to Fireworks AI / `minimax-m2p7`)
  - `model` — alias for `model1`
  - `make_model(settings)` — creates a `CentralModel` from `LlmSettings`
  - Model names and API keys are read from `MODEL1`, `MODEL2`, `FIREWORKS_API_KEY` env vars
- **`__init__.py`** — Lazily exports `model`, `model1`, `model2`, and `make_model`.

---

## Testing

```bash
# Static analysis
./cloudENV/bin/python -m compileall -q run.py dynamic_cloud model tests

# Run unit tests
./cloudENV/bin/python -m unittest discover -v

# Import verification
./cloudENV/bin/python -c "import run; import dynamic_cloud.runner; import dynamic_cloud.runtime_layers; import dynamic_cloud.workspace; print('imports ok')"
```

Tests cover:
- `test_amd_droplet.py` — Droplet lifecycle mocks
- `test_config.py` — Environment loading and settings
- `test_docker_runner.py` — SSH/SCP operation mocks
- `test_executor.py` — Execution orchestration
- `test_run_image_selection.py` — VM option lookups
- `test_runtime_dependency_checks.py` — Package scanning and filtering
- `test_runtime_validator.py` — Artifact validation

---

## Workspace Lifecycle

1. **Prepare**: `prepare_workspace()` creates `runtime_workspace/<job_id>/` with `job_spec.json`
2. **Generate**: `write_layered_payload()` populates `payload/` with all runtime files
3. **Execute**: tar pipe uploads `payload/` to the VM; results go to `outputs/<job_id>/`
4. **Shift**: `_shift_completed_outputs()` moves outputs to `final_amd/outputs/<job_id>/` and workspace to `final_amd/generated_files/<job_id>/`

---

## Cleanup

- GPU Droplets are **destroyed** after the run to stop billing (pausing does not stop billing)
- `--keep-vm` skips cleanup (billing continues)
- `--cleanup-prepared <job_dir>` destroys the Droplet for a prepared job
- If cleanup fails during an error, the Droplet ID and destroy command are printed for manual action
