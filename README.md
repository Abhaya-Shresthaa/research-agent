# Unified Orchestrator — Deep Research + Cloud Experiment on AMD GPU

> **A dual-agent system combining LLM-powered deep web research with automated ML training orchestration on AMD MI300X GPU cloud infrastructure.**

---

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [High-Level Architecture](#high-level-architecture)
- [Project Structure](#project-structure)
- [Workflow 1: Deep Research Agent](#workflow-1-deep-research-agent)
  - [Research Pipeline](#research-pipeline)
  - [Recursive Search Strategy](#recursive-search-strategy)
  - [Image Extraction & Deduplication](#image-extraction--deduplication)
  - [Report Synthesis](#report-synthesis)
  - [REST API Server](#rest-api-server)
- [Workflow 2: Cloud Experiment Agent](#workflow-2-cloud-experiment-agent)
  - [Interactive Job Builder](#interactive-job-builder)
  - [Dataset Management](#dataset-management)
  - [LLM-Generated Code Pipeline](#llm-generated-code-pipeline)
  - [Quality Gate (Static Analysis + LLM Review)](#quality-gate)
  - [AMD GPU Droplet Lifecycle](#amd-gpu-droplet-lifecycle)
  - [Remote Execution Architecture](#remote-execution-architecture)
  - [Runtime Stages](#runtime-stages)
  - [Output Helper Standard](#output-helper-standard)
- [Workflow 3: Unified Parallel Execution](#workflow-3-unified-parallel-execution)
- [Final Report Synthesis](#final-report-synthesis)
- [LLM Configuration & Providers](#llm-configuration--providers)
- [Environment Configuration](#environment-configuration)
- [SSH Key Management](#ssh-key-management)
- [Testing](#testing)
- [Security Considerations](#security-considerations)

---

## Overview

This project is a **unified orchestrator** that brings together two standalone, powerful workflows under a single CLI entry point (`main.py`):

| Agent | Purpose |
|-------|---------|
| **Deep Research Agent** | Conducts recursive, breadth-and-depth web research on any topic using LLM-powered SERP queries and Firecrawl web scraping, producing comprehensive Markdown reports with inline images and source citations. |
| **Cloud Experiment Agent** | An interactive ML training pipeline that generates training scripts via LLM, provisions AMD MI300X GPU Droplets on AMD DevCloud, runs the workload natively on the host, and downloads results — with a multi-stage quality gate. |

The two agents can run independently or together. When run together, the user answers one combined set of questions, then both agents execute in parallel and their outputs are synthesized into a unified `final-report.md`.

### Key Design Principles

1. **Dual-Mode Architecture** — Web research and GPU compute are deeply different problems. Each agent has its own tooling, lifecycle, and failure modes. The orchestrator doesn't abstract them into one thing; it sequences and composes them.
2. **LLM-for-Code with Guardrails** — The cloud workflow generates training scripts via LLM, but each script passes through a multi-stage quality gate (static AST analysis + a second LLM review) before it ever touches a GPU. This catches hallucinations, deprecated API usage, and framework mismatches before the VM is provisioned.
3. **Deterministic Infrastructure** — All VM provisioning, SSH transport, Python discovery, and runtime staging is handled by deterministic Python code — never by the LLM. The LLM only decides what ML code to write and what packages to install.
4. **Dataset Awareness** — Datasets are inspected before code generation. The LLM sees the actual directory tree, file extensions, class names, and split structure — it doesn't guess. For local datasets, inspection is entirely local (no VM needed).

### What is AMD DevCloud?

AMD DevCloud is a GPU cloud platform (compatible with the DigitalOcean API) that provides MI300X GPU instances with ROCm 7.2.4 and pre-installed ML frameworks (PyTorch 2.10.0, TensorFlow, etc.). The platform uses a droplet-like abstraction (called "GPU Droplets") that bill per-second with a 5-minute minimum.

---

## Quick Start

### Prerequisites

- Python 3.10+
- An [AMD DevCloud](https://www.amd.com/en/developer/devcloud.html) account with API access
- A [Firecrawl](https://www.firecrawl.dev/) API key for web scraping
- A [Fireworks AI](https://fireworks.ai/) API key (or any OpenAI-compatible API key)

### Setup

```bash
# 1. Clone and enter the project
cd final_amd

# 2. Create virtual environment
python3 -m venv finalENV
source finalENV/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
cd research-workflow && pip install -r requirements.txt && cd ..
cd cloud-workflow && pip install -r requirements.txt && cd ..

# 4. Configure environment
cp .env.example .env
# Edit .env with your API keys and SSH settings

# 5. Run the orchestrator
python main.py
```

### Usage

```bash
python main.py
```

Select from the menu:

```
  1. Research Agent      — Deep web research, question feedback, Markdown report
  2. Cloud Experiment    — ML job configuration, AMD VM infrastructure, remote execution
  3. Both Workflows      — Sequential questioning → parallel execution
```

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        main.py (Unified Entry Point)                │
│                                                                     │
│   ┌──────────────────────┐    ┌──────────────────────────────────┐  │
│   │  1. Research Agent    │    │   2. Cloud Experiment Agent      │  │
│   │                      │    │                                   │  │
│   │  User Question ──►   │    │  User Requirement ──►            │  │
│   │  Follow-up Q&A ──►   │    │  Framework/Accel/Dataset ──►     │  │
│   │  Deep Research ──►   │    │  LLM Questions ──► User Answers  │  │
│   │  Report/Answer ──►   │    │  VM Selection ──► Dataset Inspect│  │
│   │  outputs/report.md   │    │  LLM Generate ──► Quality Gate   │  │
│   │  outputs/answer.md   │    │  AMD Execution ──► Outputs       │  │
│   └──────────────────────┘    └──────────────────────────────────┘  │
│                                                                     │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │  3. Both Workflows (Parallel Execution)                     │  │
│   │     Sequential Q&A → Parallel Research + Cloud Training     │  │
│   │     → Final Report: outputs/final-report.md                  │  │
│   └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
final_amd/
│
├── main.py                          # Unified Orchestrator entry point
├── .env                             # Centralized environment variables
├── .env.example                     # Environment template
├── requirements.txt                 # Root dependencies
│
├── outputs/                         # Generated outputs (reports, answers)
│   ├── report.md                    # Research report
│   ├── answer.md                    # Research answer
│   ├── final-report.md              # Synthesized combined report
│   └── <job_id>/                    # Shifted cloud experiment outputs
│
├── generated_files/                 # Shifted cloud workspaces
│   └── <job_id>/
│
├── user_resources/                  # User-provided assets
│   ├── uploading_data/              # Permanent local dataset staging
│   └── user_script/                 # User-provided .py reference scripts
│
├── research-workflow/               # Deep Research Agent
│   ├── requirements.txt
│   ├── pyproject.toml
│   ├── src/
│   │   ├── api.py                   # Flask REST API server
│   │   ├── deep_research.py         # Core recursive research engine
│   │   ├── feedback.py              # Follow-up question generation
│   │   ├── prompt.py                # System prompt for research LLM
│   │   ├── run.py                   # Standalone CLI for research
│   │   ├── test_deep_research.py    # Unit tests
│   │   └── ai/
│   │       ├── providers.py         # OpenAI/Fireworks client setup
│   │       └── text_splitter.py     # Recursive text splitting
│   └── .gitignore
│
├── cloud-workflow/                  # Cloud Experiment Agent
│   ├── requirements.txt
│   ├── .env.example
│   ├── run.py                       # Interactive cloud workflow entry point
│   ├── configs/
│   │   └── job_spec.example.json
│   ├── dynamic_cloud/               # Core orchestration package
│   │   ├── amd_droplet.py           # Droplet lifecycle (DevCloud API)
│   │   ├── config.py                # Settings dataclasses + env loading
│   │   ├── dataset_config.py        # Dataset normalization + HF ID resolution
│   │   ├── docker_runner.py         # RemoteHostRunner (SSH operations)
│   │   ├── executor.py              # AMD execution + report synthesis
│   │   ├── llm_client.py            # LLM JSON helpers + robust parser
│   │   ├── llm_generator.py         # Non-interactive LLM generation
│   │   ├── payload_quality.py       # Static analyzer + review/fix prompts
│   │   ├── runtime_layers.py        # Image catalog, metadata, helpers
│   │   ├── vm_options.py            # GPU catalog, pricing, image selection
│   │   └── workspace.py             # Local workspace management
│   ├── model/
│   │   ├── __init__.py
│   │   └── model.py                 # CentralModel over OpenAI client
│   ├── runtime_workspace/           # Per-job working directories
│   ├── outputs/                     # Downloaded experiment results
│   └── tests/                       # Unit tests
│
└── finalENV/                        # Python virtual environment
```

---

## Workflow 1: Deep Research Agent

The Deep Research Agent is a Python adaptation of the "Open Deep Research" pattern. It uses an LLM to generate search queries, scrapes web content via Firecrawl, extracts learnings, and recursively deepens the investigation.

### Research Pipeline

```
User Query
    │
    ▼
[Feedback Phase] — LLM generates 3 follow-up questions
    │  User answers → combined_query = initial + Q&A
    ▼
[Deep Research Loop] — recursive breadth × depth tree
    │
    ├── Step 1: LLM generates SERP queries (breadth = N)
    │     → returns [{query, research_goal}, ...]
    │
    ├── Step 2: For each query, Firecrawl.search() with 15s timeout
    │     → returns pages with markdown content
    │
    ├── Step 3: Process SERP results via LLM
    │     → extracts learnings (up to 3 per page)
    │     → generates follow-up questions (up to 3 per page)
    │     → selects relevant images from page markdown
    │
    └── Step 4: If depth > 1, recurse with follow-up questions
          breadth' = (breadth + 1) // 2
          depth' = depth - 1
          → merges learnings and images up the call stack
```

### Recursive Search Strategy

The `deep_research()` function implements a configurable breadth × depth search tree:

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `breadth` | Number of parallel SERP queries per level | 4 | 2–10 |
| `depth` | Recursion depth of follow-up research | 2 | 1–5 |

At each recursion level:
- Breadth is halved (`(breadth + 1) // 2`) — wider at the top, narrower in follow-ups
- Learnings from all branches merged with `set()` deduplication
- URLs unioned and deduplicated
- Images pass through URL-based deduplication

### Concurrency Model

- Up to 2 concurrent Firecrawl requests (`FIRECRAWL_CONCURRENCY=2`)
- Controlled via `asyncio.Semaphore`
- Requests timeout at 15 seconds
- Failed queries return empty `ResearchResult` — they don't fail the whole research

### Image Extraction & Deduplication

1. **Candidate Extraction** — `_extract_image_candidates()` scans markdown for `![alt](url)` patterns (up to 8 per page, 40 total), capturing ±700 chars of surrounding context
2. **LLM Selection** — The `_process_serp_result()` LLM call selects genuinely relevant images with a relevance explanation
3. **Deduplication** — `_dedupe_images()` removes duplicate URLs
4. **Report Embedding** — `write_final_report()` embeds selected images inline. Images the LLM omitted from the report text are appended in a "Relevant Images" section.

### Report Synthesis

Two output modes, both using `response_format={"type": "json_object"}` with Pydantic-validated JSON:

| Mode | LLM Prompt | Output |
|------|-----------|--------|
| **Answer** | "as concise as possible — usually just a few words or maximum a sentence" | Short, focused answer |
| **Report** | "as detailed as possible, aim for 3 or more pages, include ALL the learnings" | Comprehensive report with images and sources |

### REST API Server

The research module also runs as a standalone Flask server:

```bash
cd research-workflow
python src/api.py
# Deep Research API running on port 3051
```

Endpoints:
- `POST /api/research` — runs research, returns answer + learnings + URLs + images
- `POST /api/generate-report` — runs research, returns full Markdown report

---

## Workflow 2: Cloud Experiment Agent

This is the most architecturally complex subsystem. It provisions real AMD GPU hardware, generates code with an LLM, runs it through a quality gate, and executes it on the cloud.

### Interactive Job Builder

The full interactive flow:

```
User Prompt: "train alexnet on mnist with tensorflow gpu"
    │
    ├── [Optional] User Script Check → load from user_resources/user_script/
    ├── Framework Selection (auto: PyTorch)
    ├── Accelerator Selection (auto: GPU)
    ├── Dataset Source Selection: None / HuggingFace / GitHub / Local
    ├── LLM Generates 3-5 Clarifying Questions (epochs, batch_size, LR, etc.)
    ├── User Answers Questions
    ├── VM Selection: gpu-mi300x1-192gb or gpu-mi300x8-1536gb
    ├── Droplet Image Selection: PyTorch / ROCm / vLLM / SGLang / JAX / Ubuntu
    └── Continue to dataset inspection → LLM generation → quality gate → AMD execution
```

### Dataset Management

#### Dataset Types

| Type | Configuration | Inspection |
|------|--------------|------------|
| `none` | `{type: "none"}` | Skipped |
| `local` | `{type: "local", local_paths: ["..."]}` | **Local** — no VM needed |
| `huggingface` | `{type: "huggingface", id: "ylecun/mnist"}` | Remote (VM) |
| `github` | `{type: "github", url: "..."}` | Remote (VM) |

#### Local Dataset Handling

Files are copied to `user_resources/uploading_data/` and inspected entirely locally:
- Counts files, directories, extensions
- Detects layout: split directories (train/val/test), class directories, or tabular
- Samples text files
- Returns `dataset_metadata.json` in the same format as remote inspection

#### Remote Dataset Inspection

For Hugging Face and GitHub datasets:
1. A VM is created or adopted
2. `dataset_inspector.py` runs: downloads/prepares dataset, walks tree (bounded depth 5), detects layout, inspects HF datasets
3. `dataset_metadata.json` is downloaded via SSH+Python (immune to shell banner issues)
4. The inspection Droplet is **reused** for training — the dataset cache is preserved

#### Hugging Face ID Resolution

`resolve_hf_dataset_id()` auto-resolves bare names (e.g., `"beans"`) to namespace/name (`"AI-Lab-Makerere/beans"`) via the HF Hub API. Results are cached.

#### VM-Level Dataset Cache

Downloaded external datasets are cached at `/home/<admin>/dynamic_jobs/datasets/` using a SHA-256 hash of the normalized config.

### LLM-Generated Code Pipeline

The LLM (with `GENERATION_PROMPT`) returns structured JSON:

```json
{
  "job_spec": {"job_id": "alexnet-mnist-pytorch", "objective": "...", "task_type": "training",
    "framework": "pytorch", "accelerator": "gpu", "extra_packages": ["matplotlib"],
    "dataset": {"type": "huggingface", "id": "ylecun/mnist"}},
  "files": {"generated/script.py": "import torch\n..."},
  "summary": {"task": "...", "estimated_runtime_minutes": 15, "outputs": ["model.pth"]}
}
```

**Key LLM instructions:**
- Use `outputs.py` helper functions (not hardcoded `/outputs/` paths)
- Import `ImageFolder` from `torchvision.datasets`, not `torch.utils.data`
- No `verbose=True` on LR schedulers (removed in PyTorch 2.4)
- No CUDA/NVIDIA packages
- No dataset download logic in `script.py`
- Call `save_model()` after training

### Quality Gate

The quality gate (`payload_quality.py`) is a multi-stage guard:

```
LLM Generation → Static Analyzer → LLM Review → [Fix Loop] → Pass/Reject
```

#### Static Analyzer Checks (AST-Based)

| Check | Method |
|-------|--------|
| JSON structure | Key existence |
| Script syntax | `ast.parse()` |
| Forbidden files | Set intersection (Dockerfile/requirements.txt) |
| Framework/accelerator match | `normalize_framework()` + `normalize_accelerator()` |
| Framework imports | AST import scanning |
| Epoch count | Regex + AST constant discovery and usage verification |
| Output helper imports | AST validation (auto-fixed via `_validate_outputs_imports()`) |
| `save_model()` call | Heuristics + AST for training scripts |
| Dataset download in script | String search for `load_dataset()`, `git clone` |
| TF data antipatterns | Regex + line context |
| Hallucinated imports | Full AST walk |
| Output directory writes | String search |

#### LLM Review + Repair Loop (Up to 2 Rounds)

A second LLM call reviews the payload against requirements. If issues found, the repair LLM generates a full corrected payload, which is re-analyzed.

### AMD GPU Droplet Lifecycle

The `AmdDropletManager` (`amd_droplet.py`) manages:

```
Create/Adopt → Wait for Active → Get Public IP → [Use] → Destroy
```

**Create**: POST to DevCloud API with multi-size fallback across regions.
**Adopt**: Reuse existing inspection Droplet or UI-created Droplet (`--amd-droplet-id`).
**Wait**: Poll API every 10s for `status=="active"` (600s timeout) + TCP/22 wait + SSH key login wait (300s).
**Image Selection**: Topology-aware slugs for 1× vs 8× MI300X configurations.
**Destroy**: DELETE with 3 retries. GPU Droplets bill per-second — only destroying stops billing.

### Remote Execution Architecture

Workloads run **natively on the host** (not Docker). The `RemoteHostRunner` handles:

1. **Upload**: tar-over-SSH pipe (avoids SCP fragility with shell banners)
2. **SSH Readiness**: 200s wait for basic SSH + 420s wait for vendor provisioning + banner silencing
3. **Python Discovery** (4-stage probe):
   1. System `python3`
   2. Conda environments
   3. Brute force under `/opt`, `/usr/local`, `/root`
   4. Torch module location → derive python path
   5. If no torch found → install ROCm PyTorch overlay from AMD wheel index
4. **Execution**: `python3 -u runtime_bootstrap.py` with stdout piped to `outputs/logs/runtime.log`
5. **Output Download**: fresh SSH connection (no multiplex) + `timeout`-guarded tar pipe
6. **Cleanup**: Droplet destroyed, outputs shifted to project-level directories

### Runtime Stages

`runtime_bootstrap.py` sequences:

| Stage | File | Purpose |
|-------|------|---------|
| 1 | `runtime_installer.py` | Install packages into overlay, prepare dataset |
| 2 | `runtime_runner.py` | Execute `generated/script.py` in-process via `exec()` |
| 3 | `runtime_validator.py` | Check artifact contract from `metadata.json` |
| 4 | `runtime_collector.py` | Write `run_manifest.json` |

### Output Helper Standard

Every payload includes `outputs.py` with 6 standardized functions, all resolving paths via `DYNAMIC_CLOUD_OUTPUTS_DIR`:

| Function | Writes To |
|----------|-----------|
| `save_metrics(metrics)` | `OUTPUTS/metrics.json` |
| `log_epoch(epoch, metrics)` | `OUTPUTS/training_logs.json` |
| `save_training_history(history)` | `OUTPUTS/training_logs.json` |
| `save_plot(figure, name)` | `OUTPUTS/plots/<name>` |
| `save_model(model, path, framework)` | `OUTPUTS/<path>.pth/.keras/.joblib` |
| `save_environment(extra)` | `OUTPUTS/environment.json` |

---

## Workflow 3: Unified Parallel Execution

When "Both Workflows" is selected:

1. **Single query** entered once
2. **Phase 1: Cloud questions** — framework, accelerator, dataset, clarifying questions, VM
3. **Phase 2: Research questions** — follow-ups to clarify research direction
4. **Phase 3: Parallel execution** — research runs as async task, cloud via `asyncio.to_thread()`, both gathered with `asyncio.gather(return_exceptions=True)`

---

## Final Report Synthesis

`_generate_final_report()` synthesizes research + experiment results:

1. Reads research report (`outputs/report.md`) with web images
2. Reads cloud experiment outputs (metrics, training logs, environment, runtime log, plots)
3. Passes all data as structured XML sections to an LLM with JSON-mode response format
4. LLM generates unified report covering: Executive Summary → Web Research → Experiment Results → Comparison → Key Insights → Sources
5. Post-processing: appends unembedded experiment plots and research sources

---

## LLM Configuration & Providers

### Research Workflow

| Aspect | Configuration |
|--------|--------------|
| Provider | OpenAI-compatible (Fireworks AI by default) |
| Model | `CUSTOM_MODEL` env var (default: `o3-mini`) |
| API Key | `FIREWORKS_API_KEY` or `OPENAI_KEY` |
| Context | `CONTEXT_SIZE` (default: 128K tokens) |

### Cloud Workflow

| Aspect | Configuration |
|--------|--------------|
| Provider | OpenAI-compatible (Fireworks AI) |
| Model 1 | `MODEL1` (generation, review, fix) |
| Model 2 | `MODEL2` (A/B or fallback) |
| API Key | `FIREWORKS_API_KEY` |

The `CentralModel` class wraps an `OpenAI` client with `complete_json()` and `invoke()` methods.

---

## Environment Configuration

### Key Variables

See `.env.example` for the full list. Key variables:

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `AMD_TOKEN` | **Yes** | — | AMD DevCloud API key |
| `FIREWORKS_API_KEY` | **Yes** | — | Fireworks AI API key |
| `FIRECRAWL_KEY` | **Yes** | — | Firecrawl web scraping key |
| `AMD_SSH_KEY_NAME` | **Yes** | — | SSH key name on DevCloud |
| `AMD_SSH_PRIVATE_KEY_PATH` | **Yes** | `~/.ssh/id_rsa` | Path to SSH private key |
| `MODEL1` | No | `accounts/fireworks/models/minimax-m2p7` | Primary LLM |
| `CUSTOM_MODEL` | No | `o3-mini` | Research agent LLM |
| `AMD_REGION` | No | `atl1` | DevCloud region |
| `AMD_DEFAULT_VM_SIZE` | No | `gpu-mi300x1-192gb-devcloud` | GPU plan |

Shell environment variables take highest priority and cannot be overridden by `.env`.

---

## SSH Key Management

The `AmdDropletManager` supports multiple strategies:
1. **Explicit keys** — `AMD_SSH_KEYS` env var with comma-separated key IDs/names
2. **Key lookup** — `AMD_SSH_KEY_NAME`: find by name or fingerprint
3. **Auto-registration** — `AMD_REGISTER_SSH_KEY=1`: uploads public key to DevCloud
4. **Content dedup** — checks existing keys by base64 content and MD5 fingerprint

Private key validation checks file existence and POSIX permissions (`chmod 600` required).

---

## Testing

```bash
# Research workflow tests
cd research-workflow
pip install -e .
pytest src/test_deep_research.py -v

# Cloud workflow tests
cd cloud-workflow
python -m unittest discover -v
```

---

## Security Considerations

- API keys never logged; SSH commands redacted in debug output
- NVIDIA/CUDA packages blocked from installation (would break AMD ROCm)
- Packages install into overlay directory, never modifying base system site-packages
- Droplets destroyed after each run (billing stops, attack surface removed)
- SSH: key-based auth only, VPC isolation, `StrictHostKeyChecking=accept-new`
- User-provided dataset paths use `shlex.quote()`; job IDs sanitized via regex
- Cleanup runs in `finally` blocks — executes even on failure
- Quality gate rejects payloads with CUDA packages, dataset download logic, or hallucinated imports
- Private key permissions validated before use (`chmod 600`)
- Protected environment keys cannot be overridden by `.env`
# research-agent
