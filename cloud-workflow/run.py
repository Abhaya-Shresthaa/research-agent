from __future__ import annotations

import json
import os
import shutil
import sys
import argparse
from pathlib import Path
from typing import Any

from dynamic_cloud.config import AmdDropletSession, load_amd_settings, load_environment

load_environment()

from dynamic_cloud.llm_client import generate_json, generate_json_with_model, parse_json_response
from dynamic_cloud.runtime_layers import (
    normalize_framework,
    select_image,
    validate_local_image_support,
    write_dataset_inspection_payload,
    write_layered_payload,
)
from dynamic_cloud.payload_quality import (
    FIX_PROMPT,
    REVIEW_PROMPT,
    analyze_generated_payload,
    build_fix_request,
    build_review_request,
    merge_review_notes,
    review_approved,
)
from dynamic_cloud.vm_options import (
    AMD_GPU_IMAGES,
    AMD_GPU_PRICING_PER_HOUR,
    VM_OPTIONS,
    select_amd_image,
    validate_vm_accelerator,
    vm_option_for_size,
)
from dynamic_cloud.workspace import JobWorkspace, normalize_dataset_config, prepare_workspace, safe_job_id, validate_payload

# Permanent local dataset staging directory at the base of final_amd/.
# When the user selects "Local file or folder path" as the dataset source, their
# files are copied here and inspected locally (no VM needed for inspection).
_PARENT_BASE = Path(__file__).resolve().parents[1]  # final_amd/
LOCAL_UPLOAD_DIR = _PARENT_BASE / "user_resources" / "uploading_data"
USER_SCRIPT_DIR = _PARENT_BASE / "user_resources" / "user_script"


def _check_user_script() -> tuple[bool, str]:
    """Ask if the user has their own script with requirements/methods.
    If yes, read the .py file from user_script/ and return its content.
    Returns (has_script, content).
    """
    if not _confirm("Do you have your own script with requirements/methods?", default=False):
        return False, ""

    USER_SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    script_files = sorted(USER_SCRIPT_DIR.iterdir())
    py_files = [f for f in script_files if f.suffix == ".py"]

    if not py_files:
        print(f"\n  No Python scripts found in {USER_SCRIPT_DIR}/")
        print(f"  Please place a .py script in that directory and try again.")
        return False, ""

    script_path = py_files[0]
    print(f"\n  Reading user script: {script_path}")
    content = script_path.read_text(encoding="utf-8")
    line_count = len(content.splitlines())
    print(f"  ({len(content)} bytes, {line_count} lines)")
    return True, content


# ---------------------------------------------------------------------------
# User-script prompt extensions — only appended when the user provides a script
# ---------------------------------------------------------------------------

USER_SCRIPT_QUESTION_EXTENSION = """
The user has provided their own reference script with specific requirements, architecture, or methods.
Bridge their code with the platform's capabilities when asking clarifying questions.

User's script:
```
{user_script_content}
```
"""

USER_SCRIPT_GENERATION_EXTENSION = """
CRITICAL — The user has provided their own reference script. Your generated job_spec and
generated/script.py MUST incorporate and respect the user's implementation, architecture, and
requirements from their script.

User's reference script:
```
{user_script_content}
```
"""

USER_SCRIPT_REVIEW_EXTENSION = """
When reviewing, also verify that the generated payload respects and incorporates the user's own
reference script.

User's reference script:
```
{user_script_content}
```
"""

USER_SCRIPT_FIX_EXTENSION = """
When fixing, ensure the generated script properly incorporates the user's own reference script.

User's reference script:
```
{user_script_content}
```
"""


def _format_duration(minutes: float) -> str:
    if minutes < 60:
        return f"≈ {int(minutes)} min"
    hours = minutes / 60
    if hours < 2:
        return f"≈ {hours:.1f} hr"
    return f"≈ {hours:.1f} hrs"


def _setup_and_inspect_local_dataset(selected_dataset: dict[str, Any]) -> dict[str, Any]:
    """Copy user-provided local paths to the permanent uploading_data/ directory,
    set local_paths to the individual items inside uploading_data/, and inspect
    the directory locally to produce dataset_metadata.json (no VM needed).

    Returns the metadata dict in the same format that the remote dataset
    inspector would produce.
    """
    LOCAL_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    local_paths = selected_dataset.get("local_paths") or []
    if local_paths:
        # Check if any path IS uploading_data/ itself — if so the data is
        # already staged and we just use the existing content (no copy needed).
        resolved_upload = LOCAL_UPLOAD_DIR.resolve()
        is_self_ref = any(
            Path(p).expanduser().resolve() == resolved_upload
            for p in local_paths
        )

        if not is_self_ref:
            # User provided new paths — clear uploading_data/ and re-populate
            print(f"\nCopying local data to permanent staging directory:\n  {LOCAL_UPLOAD_DIR}\n")
            for item in list(LOCAL_UPLOAD_DIR.iterdir()):
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            for raw_path in local_paths:
                source = Path(raw_path).expanduser().resolve()
                if not source.exists():
                    print(f"  [skip] Path not found: {source}")
                    continue
                target = LOCAL_UPLOAD_DIR / source.name
                if source.is_dir():
                    print(f"  Copying directory \"{source.name}\" -> uploading_data/ ...")
                    shutil.copytree(source, target)
                else:
                    print(f"  Copying file \"{source.name}\" -> uploading_data/ ...")
                    shutil.copy2(source, target)
            print()

    # Set local_paths to the individual items inside uploading_data/
    # so copy_dataset_paths copies them directly into payload/data/
    # rather than creating a nested data/uploading_data/ directory.
    contents = sorted(LOCAL_UPLOAD_DIR.iterdir())
    selected_dataset["local_paths"] = [str(p) for p in contents]
    # Clear container_data_dir so the remote dataset_manager falls through
    # to LOCAL_DATA (= WORKSPACE / "data"), which is where the tar upload
    # places the payload/data/ contents.
    selected_dataset["container_data_dir"] = ""

    # Inspect locally and return metadata
    return _inspect_local_dataset(selected_dataset, LOCAL_UPLOAD_DIR)


def _inspect_local_dataset(selected_dataset: dict[str, Any], data_path: Path) -> dict[str, Any]:
    """Scan a local dataset directory and produce metadata in the same format
    that the remote dataset inspector would produce on the VM.

    This runs entirely locally — no Droplet needed.
    """
    from collections import Counter

    dataset_path = data_path.resolve()
    if not dataset_path.exists() or not any(dataset_path.iterdir()):
        print(f"\nWarning: Local data directory is empty or missing: {dataset_path}")
        return {
            "dataset": selected_dataset,
            "dataset_path": str(dataset_path),
            "exists": False,
            "summary": {"message": "Local data directory is empty."},
            "detected_layout": {"kind": "unknown"},
        }

    files = [p for p in dataset_path.rglob("*") if p.is_file()]
    dirs = [p for p in dataset_path.rglob("*") if p.is_dir()]
    extension_counts = Counter(p.suffix.lower() or "<no_ext>" for p in files)
    top_level = sorted(p.name for p in dataset_path.iterdir())

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
    split_names = {"train", "training", "val", "valid", "validation", "test", "testing"}

    detected_layout: dict[str, Any] = {
        "kind": "unknown", "splits": {}, "class_directories": [], "label_files": [],
    }
    label_files = [
        p.relative_to(dataset_path).as_posix()
        for p in dataset_path.rglob("*")
        if p.is_file() and p.name.lower() in {"labels.csv", "train.csv", "test.csv", "metadata.csv", "annotations.csv"}
    ]
    detected_layout["label_files"] = label_files[:40]

    for child in sorted(p for p in dataset_path.iterdir() if p.is_dir()):
        lowered = child.name.lower()
        if lowered in split_names:
            class_dirs = [item.name for item in sorted(child.iterdir()) if item.is_dir()]
            image_count = sum(1 for fp in child.rglob("*") if fp.is_file() and fp.suffix.lower() in image_exts)
            detected_layout["splits"][child.name] = {
                "class_directories": class_dirs[:80],
                "class_count": len(class_dirs),
                "image_file_count": image_count,
            }

    top_class_dirs = [
        item.name for item in sorted(dataset_path.iterdir())
        if item.is_dir() and item.name.lower() not in split_names
    ]
    if detected_layout["splits"]:
        detected_layout["kind"] = "split_directory_tree"
    elif top_class_dirs:
        image_counts = {
            name: sum(1 for fp in (dataset_path / name).rglob("*") if fp.is_file() and fp.suffix.lower() in image_exts)
            for name in top_class_dirs[:80]
        }
        if any(image_counts.values()):
            detected_layout["kind"] = "class_directory_tree"
            detected_layout["class_directories"] = [
                {"name": name, "image_file_count": image_counts[name]}
                for name in top_class_dirs[:80]
            ]
    elif label_files:
        detected_layout["kind"] = "tabular_or_annotation_files"

    sample_text_exts = {".csv", ".json", ".txt", ".yaml", ".yml", ".xml", ".md"}
    samples: dict[str, Any] = {}
    sample_files = [p for p in files if p.suffix.lower() in sample_text_exts][:5]
    if sample_files:
        sample_list = []
        for sample_path in sample_files:
            try:
                text = sample_path.read_text(encoding="utf-8", errors="replace")
                lines = text.splitlines()
                sample_list.append({
                    "path": str(sample_path.relative_to(dataset_path)),
                    "size_bytes": sample_path.stat().st_size,
                    "first_lines": lines[:5],
                })
            except Exception:
                pass
        if sample_list:
            samples["text_samples"] = sample_list

    summary = {
        "root_name": dataset_path.name,
        "top_level_entries": top_level,
        "total_files": len(files),
        "total_directories": len(dirs),
        "extension_counts": dict(extension_counts.most_common(40)),
        "sample_files": [str(p.relative_to(dataset_path)) for p in files[:80]],
    }

    metadata = {
        "dataset": selected_dataset,
        "dataset_path": str(dataset_path),
        "exists": True,
        "summary": summary,
        "detected_layout": detected_layout,
        "samples": samples,
    }

    print(f"\nLocal Dataset Inspection ({dataset_path.name})")
    print("=" * 72)
    print(f"Dataset path: {dataset_path}")
    print(f"Files: {summary['total_files']}")
    print(f"Directories: {summary['total_directories']}")
    print(f"Detected layout: {detected_layout.get('kind', 'unknown')}")
    if detected_layout.get("splits"):
        for split_name, split_info in detected_layout["splits"].items():
            print(f"  - {split_name}: {split_info.get('class_count', 0)} classes, "
                  f"{split_info.get('image_file_count', 0)} images")
    elif detected_layout.get("class_directories"):
        for item in detected_layout["class_directories"][:10]:
            print(f"  - {item.get('name')}: {item.get('image_file_count', 0)} images")
    print("=" * 72)

    return metadata


def _estimate_cost(selected_vm: dict[str, Any], estimated_minutes: float | None) -> float | None:
    if estimated_minutes is None or estimated_minutes <= 0:
        return None
    vm_size = selected_vm["size"]
    hourly = AMD_GPU_PRICING_PER_HOUR.get(vm_size)
    if hourly is None:
        for pattern, rate in AMD_GPU_PRICING_PER_HOUR.items():
            if vm_size.startswith(pattern.rstrip("*")):
                hourly = rate
                break
    if hourly is None:
        return None
    return hourly * (estimated_minutes / 60)


QUESTION_PROMPT = """
You are a senior ML platform engineer.

Given the user's desired training/evaluation job, selected framework/accelerator, and selected dataset source, ask only the missing questions needed to generate a working Dockerized ML job for an AMD GPU Cloud instance.

IMPORTANT AMD ROCm CONTEXT:
- The GPU is an AMD MI300X (ROCm 7.2.4), NOT NVIDIA CUDA.
- The base Docker image already has PyTorch 2.10.0 (or TensorFlow) pre-installed with ROCm support.
- Do NOT ask about CUDA, NVIDIA, or GPU-driver-level configuration — that is handled by the platform.

Return strict JSON only:
{
  "job_title": "short safe title",
  "questions": [
    {
      "key": "snake_case_key",
      "question": "short terminal question",
      "default": "default answer or empty string",
      "why": "short reason"
    }
  ]
}

Rules:
- Ask about 5 questions when useful. Fewer is fine if the request is already specific.
- Questions must be specific to the user's request.
- Do not ask about ML framework; the program already asked that separately.
- Do not ask about runtime accelerator; the program already asked that separately.
- Do not ask about dataset source/path/link; the program already asked that separately.
- Do not ask about VM size; the program asks that separately.
- Prefer missing training details: architecture variant, split, epochs, batch size, learning rate, optimizer, metrics, output artifacts, pretrained weights, image size, augmentation, evaluation plan, or hardware-relevant settings.
- If the user already gave a detail, do not ask for it again.
- Keep defaults practical and small enough for a first AMD GPU Cloud run.
- Output only valid JSON.
"""


GENERATION_PROMPT = """
You are a senior ML engineer.

Generate only the ML job plan and training script for a layered AMD GPU Cloud Docker workflow.
Infrastructure is deterministic and handled by the Python orchestrator.

CRITICAL AMD ROCm CONTEXT (READ CAREFULLY):
- The GPU is an AMD MI300X using ROCm 7.2.4 — NOT NVIDIA CUDA.
- The base Docker image (e.g. rocm/pytorch:rocm7.2.4) ALREADY has torch, torchvision, torchaudio pre-installed with ROCm support.
- For TensorFlow/GPU, the base image has tensorflow pre-installed with ROCm support.
- Do NOT list torch, torchvision, torchaudio, tensorflow, or keras in extra_packages — they are pre-installed.
- Do NOT list any package starting with "nvidia-", "cuda", or "cupy-cuda" — these are for NVIDIA only and WILL BREAK the AMD ROCm environment.
- Use only pip packages that are pure Python or CPU-only. The GPU/ROCm support comes from the pre-built base image, not from pip packages.
- Do NOT use deprecated PyTorch arguments removed in v2.4+. In particular, do NOT pass verbose=True to ReduceLROnPlateau, StepLR, or any other optimizer scheduler — the verbose parameter was removed in PyTorch 2.4 and will crash with TypeError.
- For image folder datasets, import ImageFolder from torchvision.datasets (NOT from torch.utils.data). torch.utils.data does NOT contain ImageFolder in any modern PyTorch — it lives in torchvision.datasets.ImageFolder.

You MUST return strict JSON only. Every key below is REQUIRED — omitting any will crash the pipeline:

{
  "job_spec": {                              // **REQUIRED** — the entire job_spec object MUST be present
    "job_id": "safe-kebab-case-id",
    "objective": "...",
    "task_type": "training or evaluation or inference",
    "framework": "pytorch or tensorflow",
    "accelerator": "cpu or gpu",
    "extra_packages": ["pip packages not in base image"],
    "dataset": {
      "type": "none | local | huggingface | github",
      "id": "source identifier when applicable",
      "repo": "HF repo when applicable",
      "url": "GitHub URL when applicable",
      "local_paths": [],
      "description": "...",
      "container_data_dir": "/workspace/data for local, /workspace/prepared_datasets for remote"
    },
    "runtime": {
      "python_version": "3.11",
      "framework": "pytorch or tensorflow",
      "accelerator": "cpu or gpu",
      "entrypoint": "generated/script.py",
      "outputs_dir": "/outputs",
      "workspace_dir": "/workspace"
    },
    "expected_outputs": []
  },
  "files": {                                 // **REQUIRED** — the files object MUST be present
    "generated/script.py": "..."             // **REQUIRED** — non-empty Python script
  },
  "config": {},
  "summary": {                               // **REQUIRED** — the summary object MUST be present
    "task": "...",
    "dataset_plan": "...",
    "training_plan": "...",
    "estimated_runtime_minutes": 45,
    "outputs": ["..."],
    "notes": ["..."]
  }
}

Estimate the runtime realistically:
- Compute estimated_runtime_minutes based on epochs, dataset size, model complexity, and the AMD GPU hardware provided in the request (selected_amd_vm.size/hardware).
- AMD MI300X x1 has 1 GPU (192 GB VRAM, 20 vCPU) — budget roughly 0.5-2 min per epoch for a small CNN on a tiny dataset (MNIST/CIFAR-10 scale) using ROCm. Larger models (ResNet, ViT) or datasets (ImageNet-scale) take 5-30+ min per epoch. MI300X x8 scales near-linearly for data-parallel training.
- Default to a reasonable first-run estimate. The user can override it when executing prepared jobs.
- Do not set 45 as a default — calculate it from the actual job parameters.

Standardized output schema — ALL listed artifacts are **REQUIRED**; the generated script MUST call the appropriate save function:
  environment.json      — **REQUIRED** for every job; call save_environment()
  metrics.json          — **REQUIRED** for training/eval jobs; call save_metrics()
  training_logs.json    — **REQUIRED** for training jobs; call log_epoch() or save_training_history()
  model.pth / model.keras / model.joblib — **REQUIRED** for training jobs; call save_model() — the validation gate rejects if missing
  logs/                 — **REQUIRED**; write runtime logs here
  plots/                — **REQUIRED** for training jobs; use save_plot()

CRITICAL — paths in generated code:
- NEVER hardcode absolute paths like '/outputs/' in generated/script.py. The actual output directory is resolved at runtime via the DYNAMIC_CLOUD_OUTPUTS_DIR environment variable and may be different.
- Instead, always either:
  (a) Use the outputs.py helper functions listed below (save_metrics, log_epoch, etc.) — they resolve the correct path automatically.
  (b) Import OUTPUTS from outputs:  from outputs import OUTPUTS  then use  OUTPUTS / "logs/training.log"  (not '/outputs/logs/training.log').
  (c) Read the env var:  outputs_dir = os.environ.get('DYNAMIC_CLOUD_OUTPUTS_DIR', 'outputs')  then  os.path.join(outputs_dir, 'logs/training.log').
- A hardcoded absolute path like '/outputs/logs/training.log' will crash because that directory does not exist on the remote file system.

Hyperparameter contract:
- Put user-selected training controls near the top of generated/script.py as simple constants with literal values:
  EPOCHS = <requested integer>
  BATCH_SIZE = <requested integer>
  LEARNING_RATE = <requested float>
- Use those constants in the actual training call or loop. For TensorFlow/Keras, call model.fit(..., epochs=EPOCHS, batch_size=BATCH_SIZE). For PyTorch, loop over range(EPOCHS).
- Include the same values in save_environment(extra={"hyperparameters": ...}).
- If the user requested an epoch count, the script must visibly contain that exact integer in EPOCHS, NUM_EPOCHS, or an equivalent epochs constant.

Use the pre-bundled /workspace/generated/outputs.py helper. Only these exact functions exist — do NOT invent any others:
  save_metrics(metrics: dict) -> Path
  log_epoch(epoch: int, metrics: dict, log_file="training_logs.json") -> Path
  save_training_history(history, file_name="training_logs.json") -> Path
  save_plot(figure, name: str) -> Path  — pass plt.gcf() or any matplotlib figure; saves to plots/ automatically
  save_model(model, path="model", framework="pytorch") -> Path  — saves model.pth, returns the OUTPUTS directory (not the file path) so that os.path.join(model_path, "model.pth") resolves correctly
  save_environment(extra=None) -> Path

CRITICAL — save_model() sequence: For training jobs you MUST call save_model(model) AFTER the training loop and BEFORE any evaluation/plotting code. Without save_model(), model.pth is never written and the evaluation section crashes with FileNotFoundError. Example correct sequence:
  train_model(...)
  model_path = save_model(model)   # ← saves model.pth
  model.load_state_dict(torch.load(Path(model_path) / "model.pth"))  # ← load for eval
  evaluate_model(model)

Your generated script MUST import only from the list above. Any hallucinated import will cause a runtime error and waste AMD GPU Cloud resources.

Hard requirements:
- Output only valid JSON. No Markdown. No triple backticks.
- Do not generate a Dockerfile.
- Do not generate requirements.txt; extra packages belong in job_spec.extra_packages/runtime.extra_packages.
- Only generate generated/script.py plus structured metadata in job_spec/config.
- The orchestrator writes payload.json with dataset metadata and copies the generated payload into the pulled Docker container at /workspace, with optional local data at /workspace/data.
- The deterministic /workspace/dataset_manager.py prepares the dataset before training and exposes DATASET_PATH and DATASET_INFO_PATH environment variables.
- If dataset_metadata.json exists, runtime_bootstrap also exposes DATASET_METADATA_PATH.
- When selected_dataset_metadata is provided, use it as ground truth for dataset structure, file names, splits, class folders, feature names, and label files. Do not invent or rename folders.
- The container must write every result to the OUTPUTS directory (resolved via DYNAMIC_CLOUD_OUTPUTS_DIR env var at runtime) using the standard schema above. NEVER hardcode '/outputs/...' absolute paths in generated code.
- Put artifacts that must exist in job_spec.expected_outputs. Do not mark plots or training logs as expected for non-training jobs unless the user explicitly needs them.
- Do not assume dog_vs_cat, cats/dogs, training_set, or test_set unless explicitly requested.
- For directory datasets, derive loaders from selected_dataset_metadata.structure and selected_dataset_metadata.detected_layout.
- The LLM decides only dataset metadata, never download code. For example:
    Hugging Face -> {"type":"huggingface","id":"ylecun/mnist"}
    GitHub -> {"type":"github","url":"https://github.com/user/dataset"}
  Local -> {"type":"local","local_paths":["/absolute/local/path"],"container_data_dir":"/workspace/data"}
- generated/script.py must not call wget/curl/git clone/datasets.load_dataset/urllib download/request download logic. Use DATASET_PATH after dataset_manager prepares it.
- For Hugging Face datasets, dataset_manager saves the dataset to disk; generated/script.py may use datasets.load_from_disk(os.environ["DATASET_PATH"]).
- If dataset.type is "none", the script may use a framework built-in dataset only when the user explicitly chose that path; otherwise fail early with a clear message explaining what data is required.
- For deep learning, keep the initial run practical unless the user clearly asked for a large run.
- Include enough printed logs for the remote Docker run to be understandable.
- Use the selected framework and accelerator exactly as provided by the orchestrator.
- Use selected_dataset exactly as the dataset source metadata. Do not replace it with a different dataset.
- List only packages that are not already in common PyTorch/TensorFlow images under extra_packages.

CRITICAL: JSON string escaping rules for generated/script.py:
- The value of "generated/script.py" is a single JSON string.
- You MUST escape every double-quote inside the Python code as \" (backslash-quote).
- You MUST escape every actual newline inside the Python code as \n (backslash-n).
- You MUST escape every backslash in the Python code as \\ (double-backslash).
- In other words: the ENTIRE script content must be a valid JSON string value on ONE logical line with \\n for line breaks.
- Double-check: copy-paste your script content between "..." and verify json.loads() would parse it.
- A single unescaped double-quote, real newline, or unescaped backslash will crash the entire pipeline with a JSON decode error, wasting the GPU allocation.
"""


def main() -> None:
    args = _parse_args()
    if args.cleanup_prepared:
        _cleanup_prepared(
            Path(args.cleanup_prepared).expanduser().resolve(),
            vm_size_override=args.vm_size,
            droplet_id=args.amd_droplet_id,
        )
        return
    if args.execute_prepared:
        _execute_prepared(
            Path(args.execute_prepared).expanduser().resolve(),
            vm_size_override=args.vm_size,
            droplet_id=args.amd_droplet_id,
            droplet_ip=args.amd_droplet_ip,
        )
        return

    print("\nDynamic AMD GPU Training Job Builder\n")
    user_requirement = input("What do you want to train, test, or run?\n> ").strip()
    if not user_requirement:
        print("No requirement entered. Exiting.")
        return

    has_user_script, user_script_content = _check_user_script()

    selected_framework = _select_framework()
    selected_accelerator = _select_accelerator()
    selected_dataset = _select_dataset_source()
    print("Making personalized questions...")
    questions_payload = _ask_llm_for_questions(
        user_requirement, selected_framework, selected_accelerator, selected_dataset,
        has_user_script=has_user_script, user_script_content=user_script_content,
    )
    answers = _ask_questions(questions_payload.get("questions", []))
    selected_vm = _select_vm()
    image_slug, node_name = select_amd_image(selected_vm["size"])
    selected_vm["gpu_image"] = image_slug
    selected_vm["vm_name"] = node_name
    _apply_existing_droplet_selection(selected_vm, args.amd_droplet_id, args.amd_droplet_ip)
    validate_vm_accelerator(selected_vm, selected_accelerator)
    job_id = safe_job_id(str(questions_payload.get("job_title") or "dynamic-job"))
    dataset_metadata: dict[str, Any] | None = None
    inspection_session: AmdDropletSession | None = None
    if selected_dataset.get("type") == "local":
        # Local datasets are inspected locally — no VM needed for inspection
        dataset_metadata = _setup_and_inspect_local_dataset(selected_dataset)
    elif selected_dataset.get("type") != "none":
        dataset_metadata, selected_vm, inspection_session = _inspect_dataset_on_amd(
            job_id,
            user_requirement,
            selected_framework,
            selected_accelerator,
            selected_dataset,
            selected_vm,
        )

    generation_payload = _generate_runtime_payload(
        user_requirement,
        questions_payload,
        answers,
        selected_framework,
        selected_accelerator,
        selected_dataset,
        selected_vm,
        dataset_metadata,
        has_user_script=has_user_script,
        user_script_content=user_script_content,
    )
    job_spec = generation_payload["job_spec"]
    summary = generation_payload.get("summary", {})

    job_spec["job_id"] = job_id
    job_spec["dataset"] = normalize_dataset_config(selected_dataset or job_spec.get("dataset") or {})
    job_spec["framework"] = selected_framework
    job_spec["accelerator"] = selected_accelerator
    job_spec["amd_vm_size"] = selected_vm["size"]
    job_spec["amd_vm_note"] = selected_vm["cost_note"]
    job_spec["amd_gpu_image"] = selected_vm.get("gpu_image", "")
    runtime = job_spec.setdefault("runtime", {})
    runtime["framework"] = selected_framework
    runtime["accelerator"] = selected_accelerator
    runtime["image"] = select_image(selected_framework, selected_accelerator)
    validate_local_image_support(selected_framework, selected_accelerator)

    workspace = prepare_workspace(job_spec, reset=dataset_metadata is None)
    if not dataset_metadata and workspace.payload_dir.joinpath("dataset_metadata.json").exists():
        try:
            dataset_metadata = json.loads(workspace.payload_dir.joinpath("dataset_metadata.json").read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    job_spec["job_id"] = workspace.job_id
    estimated_minutes = summary.get("estimated_runtime_minutes")
    if estimated_minutes:
        job_spec["estimated_runtime_minutes"] = estimated_minutes
    if dataset_metadata:
        job_spec["dataset_metadata_path"] = "dataset_metadata.json"
        config = generation_payload.setdefault("config", {})
        if isinstance(config, dict):
            config["dataset_metadata"] = dataset_metadata
    metadata = write_layered_payload(workspace, generation_payload)
    if dataset_metadata:
        workspace.payload_dir.joinpath("dataset_metadata.json").write_text(json.dumps(dataset_metadata, indent=2), encoding="utf-8")
    validate_payload(workspace)

    _print_summary(job_spec, summary, selected_vm, workspace, metadata)

    try:
        _load_selected_amd_settings(workspace, selected_vm)
    except Exception as exc:
        print("\nAMD Cloud setup is not ready yet:")
        print(f"  {exc}")
        print("\nPrepared files were still written here:")
        print(f"  {workspace.payload_dir}")
        print("\nFix the AMD configuration, then run this script again.")
        return

    estimated_minutes = job_spec.get("estimated_runtime_minutes")
    if estimated_minutes:
        estimated_cost = _estimate_cost(selected_vm, estimated_minutes)
        print(f"\nEstimated runtime: {_format_duration(estimated_minutes)}")
        if estimated_cost is not None:
            print(f"Estimated AMD GPU cost: ${estimated_cost:.2f}")

    # Determine whether a VM already exists from a prior inspection stage.
    # For "none" and "local" dataset types, no VM was created yet.
    vm_already_running = inspection_session is not None

    run_prompt = (
        "Continue on the same AMD Droplet and run training now?"
        if vm_already_running
        else "Start AMD GPU Droplet and run training now?"
    )
    if not _confirm(run_prompt):
        print("\nPrepared files only. You can inspect them here:")
        print(f"  {workspace.payload_dir}")
        print(f"\nTo execute later run:\n  python run.py --execute-prepared {workspace.root}")
        if inspection_session:
            _cleanup_selected_vm(workspace, selected_vm, inspection_session)
        return

    keep_vm = bool(
        inspection_session and not inspection_session.created_for_dataset_inspection
    ) or (
        False
        if vm_already_running
        else _confirm("Keep the Droplet after the run for debugging? (billing continues)", default=False)
    )
    _execute_on_amd(
        workspace,
        job_spec,
        selected_vm,
        keep_vm,
        preserve_remote=vm_already_running,
        vm_already_selected=vm_already_running,
        session=inspection_session,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive dynamic AMD GPU training runner.")
    parser.add_argument(
        "--execute-prepared",
        help="Run an already prepared job folder or payload folder without regenerating files.",
    )
    parser.add_argument(
        "--cleanup-prepared",
        help="Destroy AMD Droplet for an already prepared job folder or payload folder.",
    )
    parser.add_argument(
        "--vm-size",
        help="Override the AMD GPU plan for --execute-prepared, for example gpu-mi300x8-1536gb-devcloud.",
    )
    parser.add_argument(
        "--amd-droplet-id",
        default=os.getenv("AMD_EXISTING_DROPLET_ID") or None,
        help="Reuse an existing AMD GPU Droplet ID instead of creating one through the API.",
    )
    parser.add_argument(
        "--amd-droplet-ip",
        default=os.getenv("AMD_EXISTING_DROPLET_IP") or None,
        help="Public IPv4 for --amd-droplet-id. If omitted, the runner looks it up from the API.",
    )
    return parser.parse_args()


def _ask_llm_for_questions(
    user_requirement: str,
    selected_framework: str,
    selected_accelerator: str,
    selected_dataset: dict[str, Any],
    has_user_script: bool = False,
    user_script_content: str = "",
) -> dict[str, Any]:
    request = {
        "user_requirement": user_requirement,
        "selected_framework": selected_framework,
        "selected_accelerator": selected_accelerator,
        "selected_dataset": selected_dataset,
    }

    prompt = QUESTION_PROMPT
    if has_user_script and user_script_content:
        prompt += USER_SCRIPT_QUESTION_EXTENSION.format(user_script_content=user_script_content)

    payload = generate_json(prompt, request)
    questions = payload.get("questions")
    if not isinstance(questions, list) or not questions:
        raise RuntimeError("The model did not return clarification questions.")
    return payload


def _ask_questions(questions: list[dict[str, Any]]) -> dict[str, str]:
    print("\nA few details are needed before generating the job:\n")
    answers: dict[str, str] = {}
    for index, item in enumerate(questions[:5], start=1):
        key = str(item.get("key") or f"answer_{index}")
        question = str(item.get("question") or f"Detail {index}?")
        default = str(item.get("default") or "").strip()
        suffix = f" [{default}]" if default else ""
        answer = input(f"{index}. {question}{suffix}\n> ").strip()
        answers[key] = answer or default
    return answers


def _select_framework() -> str:
    print("\n" + "-" * 59)
    print("\nChoose the ML framework for this run.\n"
          "PyTorch is better optimized on AMD GPU with prebuilt base image.\n")
    print("Selected: PyTorch")
    return "pytorch"


def _select_accelerator() -> str:
    print("\nRuntime accelerator: GPU (AMD MI300X)")
    input("\nPress Enter to continue...\n")
    return "gpu"


def _select_dataset_source() -> dict[str, Any]:
    options = [
        ("none", "No external dataset / framework built-in"),
        ("huggingface", "Hugging Face dataset ID"),
        ("github", "GitHub repository or archive URL"),
        ("local", "Local file or folder path"),
    ]
    print("\n" + "-" * 59)
    print("\nChoose the dataset source for this run.\n")
    for index, (_value, label) in enumerate(options, start=1):
        print(f"{index}. {label}")

    while True:
        choice = input("\nEnter dataset source option number (default: 1 - no external dataset)\n> ").strip()
        if not choice:
            selected_type = "none"
            break
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            selected_type = options[int(choice) - 1][0]
            break
        normalized = str(choice).strip().lower().replace("-", "_")
        aliases = {
            "hf": "huggingface",
            "hugging_face": "huggingface",
            "builtin": "none",
            "built_in": "none",
        }
        selected_type = aliases.get(normalized, normalized)
        if selected_type in {value for value, _label in options}:
            break
        print(f"Please choose a number from 1 to {len(options)}.")

    config: dict[str, Any] = {"type": selected_type}
    if selected_type == "none":
        print("Selected: no external dataset")
        return normalize_dataset_config(config)

    prompts = {
        "huggingface": "Enter Hugging Face dataset ID, for example AI-Lab-Makerere/beans",
        "github": "Enter GitHub repo/archive URL",
        "local": "Enter local dataset path(s), comma-separated if multiple",
    }
    value = input(f"\n{prompts[selected_type]}\n> ").strip()
    if not value:
        if selected_type == "local" and LOCAL_UPLOAD_DIR.exists() and any(LOCAL_UPLOAD_DIR.iterdir()):
            print("No path entered; using existing data in uploading_data/.")
            config["local_paths"] = [str(LOCAL_UPLOAD_DIR)]
        else:
            print("No dataset identifier entered; using no external dataset.")
            return normalize_dataset_config({"type": "none"})
    elif selected_type == "local":
        config["local_paths"] = [item.strip() for item in value.split(",") if item.strip()]
    elif selected_type == "github":
        config["url"] = value
    else:
        config["id"] = value
        if selected_type == "huggingface":
            config["repo"] = value

    normalized = normalize_dataset_config(config)
    print(f"Selected dataset source: {normalized.get('type')}")
    return normalized


def _select_vm() -> dict[str, str]:
    print("\n" + "-" * 59)
    print("\nChoose the AMD GPU Droplet for this run.")
    print("All GPU Droplets are billed at $1.99/GPU/hr and destroyed after the run.\n")
    for index, option in enumerate(VM_OPTIONS, start=1):
        print(f"{index}. {option['size']} - {option['label']}")
        print(f"   {option['hardware']}")
        if option.get("storage"):
            print(f"   {option['storage']}")
        print(f"   {option['cost_note']}")

    while True:
        choice = input(f"\nEnter GPU plan number (default: 1 - {VM_OPTIONS[0]['size']})\n> ").strip()
        if not choice:
            chosen = VM_OPTIONS[0]
            print(f"Selected: 1 - {chosen['size']} ({chosen['label']})")
            return chosen
        if choice.isdigit() and 1 <= int(choice) <= len(VM_OPTIONS):
            chosen = VM_OPTIONS[int(choice) - 1]
            print(f"Selected: {int(choice)} - {chosen['size']} ({chosen['label']})")
            return chosen
        print(f"Please choose a number from 1 to {len(VM_OPTIONS)}.")


def _generate_runtime_payload(
    user_requirement: str,
    questions_payload: dict[str, Any],
    answers: dict[str, str],
    selected_framework: str,
    selected_accelerator: str,
    selected_dataset: dict[str, Any],
    selected_vm: dict[str, str],
    dataset_metadata: dict[str, Any] | None = None,
    has_user_script: bool = False,
    user_script_content: str = "",
) -> dict[str, Any]:
    request = {
        "user_requirement": user_requirement,
        "clarifying_questions": questions_payload,
        "answers": answers,
        "selected_framework": selected_framework,
        "selected_accelerator": selected_accelerator,
        "selected_dataset": selected_dataset,
        "selected_dataset_metadata": dataset_metadata,
        "selected_runtime_image": select_image(selected_framework, selected_accelerator),
        "selected_amd_vm": selected_vm,
    }

    if has_user_script and user_script_content:
        request["user_script"] = user_script_content

    print("\nGenerating structured job spec and script.py...")

    generation_prompt = GENERATION_PROMPT
    if has_user_script and user_script_content:
        generation_prompt += USER_SCRIPT_GENERATION_EXTENSION.format(user_script_content=user_script_content)

    payload = generate_json(generation_prompt, request)
    _validate_generation_payload(payload)
    payload = _review_and_fix_runtime_payload(payload, request, has_user_script=has_user_script, user_script_content=user_script_content)
    return payload


def _inspect_dataset_on_amd(
    job_id: str,
    user_requirement: str,
    selected_framework: str,
    selected_accelerator: str,
    selected_dataset: dict[str, Any],
    selected_vm: dict[str, str],
) -> tuple[dict[str, Any], dict[str, str], AmdDropletSession]:
    from dynamic_cloud.amd_droplet import AmdDropletManager
    from dynamic_cloud.docker_runner import RemoteHostRunner

    inspect_spec = {
        "job_id": job_id,
        "objective": user_requirement,
        "framework": selected_framework,
        "accelerator": selected_accelerator,
        "dataset": normalize_dataset_config(selected_dataset),
        "amd_vm_size": selected_vm["size"],
    }
    workspace = prepare_workspace(inspect_spec, reset=True)
    metadata = write_dataset_inspection_payload(workspace, inspect_spec, selected_framework, selected_accelerator)
    amd_settings = _load_selected_amd_settings(workspace, selected_vm)
    droplet = AmdDropletManager(amd_settings)
    actual_selected_vm = dict(selected_vm)
    actual_selected_vm["vm_name"] = amd_settings.vm_name
    host = RemoteHostRunner(amd_settings)
    existing_droplet_id = _selected_existing_droplet_id(selected_vm)
    existing_droplet_ip = str(selected_vm.get("existing_droplet_ip") or "").strip()

    print("\nDataset Inspection")
    print("=" * 72)
    print(f"Job ID: {workspace.job_id}")
    print(f"Dataset source: {selected_dataset.get('type')}")
    print(f"GPU Plan: {selected_vm['size']} - {selected_vm['label']}")
    print(f"AMD Droplet image: {amd_settings.gpu_image}")
    print(f"Selected Droplet: {amd_settings.vm_name}")
    if existing_droplet_id:
        print(f"Using existing AMD Droplet ID: {existing_droplet_id}")
    print("The Droplet will stay available for the training run so the downloaded dataset can be reused.")
    print("=" * 72)

    prompt = (
        "Use the existing AMD GPU Droplet now to download and inspect the dataset?"
        if existing_droplet_id
        else "Start AMD GPU Droplet now to download and inspect the dataset?"
    )
    if not _confirm(prompt, default=True):
        raise RuntimeError("Dataset inspection is required before script generation for external datasets.")

    try:
        if existing_droplet_id:
            droplet.adopt(existing_droplet_id)
            if existing_droplet_ip:
                ip = existing_droplet_ip
            else:
                droplet.wait_until_active()
                ip = droplet.get_public_ip()
            print("Checking SSH access on the selected Droplet...")
            droplet.wait_for_ssh(ip)
        else:
            droplet.create_droplet()
            droplet.wait_until_active()
            ip = droplet.get_public_ip()
        print(f"Droplet public IP: {ip}")
        remote_job_dir = host.upload_workspace(ip, workspace)
        host.run_dataset_inspection(ip, remote_job_dir)
        metadata_path = host.download_dataset_metadata(ip, workspace, remote_job_dir)
        dataset_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        _print_dataset_metadata_summary(dataset_metadata)
        if not droplet.droplet_id:
            raise RuntimeError("Dataset inspection completed but no Droplet ID was recorded.")
        session = AmdDropletSession(
            droplet_id=droplet.droplet_id,
            ip=ip,
            vm_name=amd_settings.vm_name,
            remote_job_dir=remote_job_dir,
            created_for_dataset_inspection=not bool(existing_droplet_id),
        )
        return dataset_metadata, actual_selected_vm, session
    except BaseException:
        if existing_droplet_id:
            print("Dataset inspection failed; keeping existing AMD Droplet.")
        else:
            print("Dataset inspection failed; destroying AMD Droplet.")
            try:
                droplet.destroy()
            except BaseException as cleanup_exc:
                print(f"Cleanup failed: {cleanup_exc}")
                if droplet.droplet_id:
                    print(f"IMPORTANT: Manually destroy Droplet {droplet.droplet_id} to stop billing!")
        raise


def _review_and_fix_runtime_payload(
    payload: dict[str, Any],
    requirement_context: dict[str, Any],
    max_fix_attempts: int = 2,
    has_user_script: bool = False,
    user_script_content: str = "",
) -> dict[str, Any]:
    from model import model1

    print("Running static analyzer and LLM code review...")

    review_prompt = REVIEW_PROMPT
    fix_prompt = FIX_PROMPT
    if has_user_script and user_script_content:
        review_prompt += USER_SCRIPT_REVIEW_EXTENSION.format(user_script_content=user_script_content)
        fix_prompt += USER_SCRIPT_FIX_EXTENSION.format(user_script_content=user_script_content)

    for attempt in range(max_fix_attempts + 1):
        static_report = analyze_generated_payload(payload, requirement_context)
        review_report = generate_json_with_model(
            model1,
            review_prompt,
            build_review_request(requirement_context, payload, static_report),
        )

        if review_approved(static_report, review_report):
            payload["quality_gate"] = {
                "static_analyzer": static_report,
                "llm_review": review_report,
                "fix_attempts": attempt,
            }
            if attempt:
                print(f"Quality gate passed after {attempt} repair attempt(s).")
            else:
                print("Quality gate passed.")
            return payload

        if attempt >= max_fix_attempts:
            notes = "\n".join(merge_review_notes(static_report, review_report))
            raise RuntimeError(f"Generated payload failed quality review:\n{notes}")

        print(f"Quality gate found issues; asking LLM to repair payload (attempt {attempt + 1}).")
        payload = generate_json_with_model(
            model1,
            fix_prompt,
            build_fix_request(requirement_context, payload, static_report, review_report),
        )
        _validate_generation_payload(payload)

    raise RuntimeError("Unreachable payload quality loop exit.")


def _validate_generation_payload(payload: dict[str, Any]) -> None:
    if not isinstance(payload.get("job_spec"), dict):
        raise RuntimeError("Generated payload is missing job_spec.")
    files = payload.get("files")
    if not isinstance(files, dict):
        raise RuntimeError("Generated payload is missing files.")
    required_files = {"generated/script.py"}
    missing = required_files - set(files)
    if missing:
        raise RuntimeError(f"Generated payload missing files: {', '.join(sorted(missing))}")


def _parse_json_response(content: str) -> dict[str, Any]:
    return parse_json_response(content)


def _print_summary(
    job_spec: dict[str, Any],
    summary: dict[str, Any],
    selected_vm: dict[str, str],
    workspace: Any,
    metadata: dict[str, Any] | None = None,
) -> None:
    print("\nGenerated Job Summary")
    print("=" * 72)
    print(f"Job ID: {job_spec.get('job_id')}")
    print(f"Task: {summary.get('task') or job_spec.get('objective')}")
    print(f"Dataset: {summary.get('dataset_plan') or job_spec.get('dataset', {}).get('description')}")
    print(f"Training: {summary.get('training_plan') or 'Generated in script.py'}")
    if metadata:
        print(f"Framework: {metadata.get('framework')} ({metadata.get('accelerator')})")
        print(f"Runtime image: {metadata.get('image')}")
        packages = metadata.get("extra_packages") or []
        print(f"Runtime packages: {', '.join(packages) if packages else 'none'}")
    print(f"GPU Plan: {selected_vm['size']} - {selected_vm['label']}")
    print(f"Plan note: {selected_vm['cost_note']}")
    print(f"Payload folder: {workspace.payload_dir}")
    print(f"Outputs folder after run: cloud-workflow/outputs/{workspace.job_id}")

    outputs = summary.get("outputs") or job_spec.get("expected_outputs") or []
    if outputs:
        print("\nExpected outputs:")
        for output in outputs:
            print(f"  - {output}")

    notes = summary.get("notes") or []
    if notes:
        print("\nNotes:")
        for note in notes:
            print(f"  - {note}")
    print("=" * 72)


def _print_dataset_metadata_summary(dataset_metadata: dict[str, Any]) -> None:
    summary = dataset_metadata.get("summary") if isinstance(dataset_metadata.get("summary"), dict) else {}
    layout = dataset_metadata.get("detected_layout") if isinstance(dataset_metadata.get("detected_layout"), dict) else {}
    print("\nDataset Metadata Summary")
    print("=" * 72)
    print(f"Dataset path: {dataset_metadata.get('dataset_path')}")
    print(f"Root name: {summary.get('root_name')}")
    print(f"Files: {summary.get('total_files', 0)}")
    print(f"Directories: {summary.get('total_directories', 0)}")
    print(f"Detected layout: {layout.get('kind', 'unknown')}")
    if layout.get("splits"):
        print("Splits:")
        for split_name, split_info in layout["splits"].items():
            print(f"  - {split_name}: {split_info.get('class_count', 0)} classes, {split_info.get('image_file_count', 0)} images")
    elif layout.get("class_directories"):
        print("Class directories:")
        for item in layout["class_directories"][:20]:
            print(f"  - {item.get('name')}: {item.get('image_file_count', 0)} images")
    extensions = summary.get("extension_counts") or {}
    if extensions:
        print("File extensions:")
        for extension, count in list(extensions.items())[:12]:
            print(f"  - {extension}: {count}")
    print("=" * 72)


def _confirm(prompt: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        answer = input(f"\n{prompt} {suffix}\n> ").strip().lower()
        if not answer:
            print(f"Selected: {'yes' if default else 'no'}")
            return default
        if answer in {"y", "yes"}:
            print("Selected: yes")
            return True
        if answer in {"n", "no"}:
            print("Selected: no")
            return False
        print("Please enter y/yes or n/no.")


def _execute_on_amd(
    workspace: Any,
    job_spec: dict[str, Any],
    selected_vm: dict[str, str],
    keep_vm: bool,
    preserve_remote: bool = False,
    vm_already_selected: bool = False,
    session: AmdDropletSession | None = None,
) -> None:
    from dynamic_cloud.executor import execute_workspace_on_amd

    amd_settings = _load_selected_amd_settings(workspace, selected_vm)
    validate_vm_accelerator(selected_vm, str((job_spec.get("runtime") or {}).get("accelerator") or job_spec.get("accelerator") or "gpu"))
    execute_workspace_on_amd(
        workspace,
        job_spec,
        amd_settings,
        keep_vm,
        preserve_remote=preserve_remote,
        vm_already_selected=vm_already_selected,
        session=session,
    )


def _cleanup_selected_vm(
    workspace: Any,
    selected_vm: dict[str, str],
    session: AmdDropletSession | None = None,
) -> None:
    from dynamic_cloud.amd_droplet import AmdDropletManager

    try:
        if session and not session.created_for_dataset_inspection:
            print("\nKeeping existing AMD Droplet because it was not created by this workflow.")
            return
        print("\nDestroying the inspection Droplet because training was not started.")
        manager = AmdDropletManager(_load_selected_amd_settings(workspace, selected_vm))
        if session:
            manager.adopt(session.droplet_id)
        manager.destroy()
    except BaseException as exc:
        print(f"Cleanup failed: {exc}")


def _execute_prepared(
    path: Path,
    vm_size_override: str | None = None,
    droplet_id: str | None = None,
    droplet_ip: str | None = None,
) -> None:
    workspace, job_spec = _load_prepared_workspace(path)
    validate_payload(workspace)

    saved_vm_size = str(job_spec.get("amd_vm_size") or VM_OPTIONS[0]["size"])
    vm_size = vm_size_override or saved_vm_size
    if vm_size_override and vm_size_override != saved_vm_size:
        print(f"Warning: overriding saved GPU plan {saved_vm_size} with {vm_size_override}.")
    job_spec["amd_vm_size"] = vm_size
    selected_vm = _vm_option_for_size(vm_size, job_spec)
    saved_gpu_image = str(job_spec.get("amd_gpu_image") or "").strip()
    if saved_gpu_image:
        selected_vm["gpu_image"] = saved_gpu_image
    else:
        selected_vm["gpu_image"] = _amd_droplet_image_for_framework(
            str((job_spec.get("runtime") or {}).get("framework") or job_spec.get("framework") or "pytorch")
        )
    _apply_existing_droplet_selection(selected_vm, droplet_id, droplet_ip)
    validate_vm_accelerator(selected_vm, str((job_spec.get("runtime") or {}).get("accelerator") or job_spec.get("accelerator") or "gpu"))

    print("\nPrepared Job Execution")
    print("=" * 72)
    print(f"Job ID: {workspace.job_id}")
    print(f"Payload folder: {workspace.payload_dir}")
    print(f"GPU Plan: {selected_vm['size']} - {selected_vm['label']}")

    print("=" * 72)

    estimated_minutes = job_spec.get("estimated_runtime_minutes")
    if estimated_minutes:
        estimated_cost = _estimate_cost(selected_vm, estimated_minutes)
        print(f"Estimated runtime: {_format_duration(estimated_minutes)}")
        if estimated_cost is not None:
            print(f"Estimated AMD GPU cost: ${estimated_cost:.2f}")

    existing_droplet_id = _selected_existing_droplet_id(selected_vm)
    prompt = (
        "Run this prepared job on the existing AMD GPU Droplet now?"
        if existing_droplet_id
        else "Start AMD GPU Droplet and run this prepared job now?"
    )
    if not _confirm(prompt):
        print("Stopped before AMD execution.")
        print(f"\nTo execute later run:\n  python run.py --execute-prepared {workspace.root}")
        return

    session = _existing_droplet_session(workspace, selected_vm) if existing_droplet_id else None
    keep_vm = True if session else _confirm("Keep the Droplet after the run for debugging? (billing continues)", default=False)
    _execute_on_amd(
        workspace,
        job_spec,
        selected_vm,
        keep_vm,
        vm_already_selected=bool(session),
        session=session,
    )


def _cleanup_prepared(
    path: Path,
    vm_size_override: str | None = None,
    droplet_id: str | None = None,
) -> None:
    from dynamic_cloud.amd_droplet import AmdDropletManager

    workspace, job_spec = _load_prepared_workspace(path)
    vm_size = vm_size_override or str(job_spec.get("amd_vm_size") or VM_OPTIONS[0]["size"])
    selected_vm = _vm_option_for_size(vm_size, job_spec)
    amd_settings = _load_selected_amd_settings(workspace, selected_vm)
    print(f"Destroying AMD Droplet for prepared job: {workspace.job_id}")
    manager = AmdDropletManager(amd_settings)
    resolved_droplet_id = (droplet_id or os.getenv("AMD_EXISTING_DROPLET_ID") or "").strip()
    if not resolved_droplet_id:
        raise RuntimeError(
            "No Droplet ID was provided for cleanup. Use --amd-droplet-id or set AMD_EXISTING_DROPLET_ID."
        )
    if not resolved_droplet_id.isdigit():
        raise ValueError(f"AMD Droplet ID must be numeric, got {resolved_droplet_id!r}.")
    manager.adopt(int(resolved_droplet_id))
    manager.destroy()


def _load_prepared_workspace(path: Path) -> tuple[JobWorkspace, dict[str, Any]]:
    payload_dir = path if path.name == "payload" else path / "payload"
    root = payload_dir.parent
    job_spec_path = root / "job_spec.json"
    if not payload_dir.exists():
        raise FileNotFoundError(f"Prepared payload folder not found: {payload_dir}")
    if not job_spec_path.exists():
        raise FileNotFoundError(f"Prepared job spec not found: {job_spec_path}")

    job_spec = json.loads(job_spec_path.read_text(encoding="utf-8"))
    workspace = JobWorkspace(
        job_id=root.name,
        root=root,
        payload_dir=payload_dir,
        generated_dir=payload_dir / "generated",
        data_dir=payload_dir / "data",
    )
    return workspace, job_spec


def _vm_option_for_size(vm_size: str, job_spec: dict[str, Any] | None = None) -> dict[str, Any]:
    return vm_option_for_size(vm_size, job_spec)


def _load_selected_amd_settings(workspace: Any, selected_vm: dict[str, str]) -> Any:
    from dataclasses import replace

    vm_name = selected_vm.get("vm_name") or workspace.job_id[:55]
    amd_settings = load_amd_settings(vm_name=vm_name)
    return replace(
        amd_settings,
        vm_sizes=(selected_vm["size"],),
        gpu_image=selected_vm.get("gpu_image") or amd_settings.gpu_image,
    )


def _apply_existing_droplet_selection(
    selected_vm: dict[str, Any],
    droplet_id: str | None,
    droplet_ip: str | None,
) -> None:
    resolved_id = (droplet_id or os.getenv("AMD_EXISTING_DROPLET_ID") or "").strip()
    resolved_ip = (droplet_ip or os.getenv("AMD_EXISTING_DROPLET_IP") or "").strip()
    if resolved_id:
        selected_vm["existing_droplet_id"] = resolved_id
    if resolved_ip:
        selected_vm["existing_droplet_ip"] = resolved_ip


def _selected_existing_droplet_id(selected_vm: dict[str, Any]) -> int | None:
    raw_value = str(selected_vm.get("existing_droplet_id") or "").strip()
    if not raw_value:
        return None
    if not raw_value.isdigit():
        raise ValueError(f"AMD Droplet ID must be numeric, got {raw_value!r}.")
    return int(raw_value)


def _existing_droplet_session(workspace: Any, selected_vm: dict[str, Any]) -> AmdDropletSession:
    from dynamic_cloud.amd_droplet import AmdDropletManager

    droplet_id = _selected_existing_droplet_id(selected_vm)
    if not droplet_id:
        raise RuntimeError("No existing AMD Droplet ID was selected.")
    amd_settings = _load_selected_amd_settings(workspace, selected_vm)
    manager = AmdDropletManager(amd_settings)
    manager.adopt(droplet_id)
    ip = str(selected_vm.get("existing_droplet_ip") or "").strip()
    if not ip:
        manager.wait_until_active()
        ip = manager.get_public_ip()
    else:
        print("Checking SSH access on the selected Droplet...")
        manager.wait_for_ssh(ip)
    return AmdDropletSession(
        droplet_id=droplet_id,
        ip=ip,
        vm_name=amd_settings.vm_name,
        created_for_dataset_inspection=False,
    )


def _amd_droplet_image_for_framework(framework: str) -> str:
    normalized = normalize_framework(framework)
    env_key = f"AMD_{normalized.upper()}_GPU_IMAGE"
    explicit = (os.getenv(env_key) or "").strip()
    if explicit:
        return explicit
    legacy = (os.getenv("AMD_GPU_IMAGE") or "").strip()
    if legacy:
        return legacy
    return AMD_GPU_IMAGES.get(normalized, AMD_GPU_IMAGES["rocm"])["slug"]


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
