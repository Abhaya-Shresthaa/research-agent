from __future__ import annotations

import ast
import inspect
import json
import os
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from dynamic_cloud.dataset_config import normalize_dataset_config, resolve_hf_dataset_id
from dynamic_cloud.workspace import JobWorkspace


IMAGE_CATALOG = {
    ("pytorch", "cpu"): os.getenv("PYTORCH_CPU_IMAGE", "abhaya123/pytorch-cpu:latest"),
    ("tensorflow", "cpu"): os.getenv("TENSORFLOW_CPU_IMAGE", "abhaya123/tensorflow-cpu:latest"),
    ("pytorch", "gpu"): os.getenv("PYTORCH_GPU_IMAGE", "rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_2.10.0"),
    ("tensorflow", "gpu"): os.getenv("TENSORFLOW_GPU_IMAGE", "rocm/tensorflow:rocm7.2.4"),
}

SUPPORTED_LOCAL_IMAGES = {
    ("pytorch", "cpu"),
    ("tensorflow", "cpu"),
}

BASE_IMPORTS = {
    # Python stdlib — guaranteed on every Python installation
    "json",
    "__future__",
    "os",
    "pathlib",
    "random",
    "re",
    "sys",
    "time",
    "typing",
    # Workflow-internal modules (not third-party packages)
    "outputs",
    "generated",
    "dataset_manager",
}

# Third-party packages *not* included above are detected by
# ``missing_runtime_packages()`` and installed via pip into the overlay.
# Core DL frameworks (torch, tensorflow, etc.) are excluded by
# ``_filter_preinstalled_frameworks()`` via CORE_DL_PACKAGES below.

IMPORT_TO_PACKAGE = {
    "PIL": "pillow",
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
    "yaml": "pyyaml",
}

PACKAGE_TO_IMPORT = {
    "opencv-python": "cv2",
    "scikit-learn": "sklearn",
    "pillow": "PIL",
    "pyyaml": "yaml",
    "azure-storage-blob": "azure.storage.blob",
}

PACKAGE_REQUIRED_ATTRIBUTES = {
    "azure-storage-blob": ["BlobServiceClient"],
    "datasets": ["load_dataset", "load_from_disk"],
}

# Core DL frameworks pre-installed on AMD Quick Start Images — never try to
# upgrade or reinstall these to avoid breaking the host's ROCm integration.
CORE_DL_PACKAGES = {
    "torch", "torchvision", "torchaudio",
    "pytorch", "pytorch-*",
    "tensorflow", "keras", "tf", "tensorflow-gpu",
    "jax", "jaxlib",
    "rocm", "rocm-*",
    "amd", "amd-*",
}

# These packages are for CUDA/NVIDIA images, not AMD ROCm images.  They must
# never be introduced by generated metadata: apart from wasting several GB,
# installing them can make pip try to replace packages supplied by the AMD
# Quick Start image, and for some packages pip will try to uninstall a
# Debian-installed package whose RECORD file does not exist.
AMD_PROTECTED_PACKAGE_PREFIXES = (
    "nvidia-",
    "cuda",
    "cupy-cuda",
    "nvcc",
    "cudnn",
    "cublas",
    "nccl",
)

# Additional specific CUDA/NVIDIA package names that must be blocked outright.
AMD_BLOCKED_PACKAGES = {
    "nvidia-ml-py3",
    "pycuda",
    "torch-cuda",
    "tf-nightly",
    "tensorflow-cpu",
    "tensorflow-gpu",
}


def normalize_framework(value: str) -> str:
    text = value.strip().lower()
    aliases = {
        "torch": "pytorch",
        "py torch": "pytorch",
        "tf": "tensorflow",
        "tensor flow": "tensorflow",
    }
    return aliases.get(text, text)


def normalize_accelerator(value: str) -> str:
    text = value.strip().lower()
    if text in {"cuda", "gpu", "nvidia", "rocm", "amd", "mi300x"}:
        return "gpu"
    return "cpu"


def select_image(framework: str, accelerator: str) -> str:
    key = (normalize_framework(framework), normalize_accelerator(accelerator))
    image = IMAGE_CATALOG.get(key)
    if not image:
        supported = ", ".join(f"{name}/{device}" for name, device in sorted(IMAGE_CATALOG))
        raise ValueError(f"Unsupported runtime {key[0]}/{key[1]}. Supported: {supported}")
    return image


def validate_local_image_support(framework: str, accelerator: str) -> None:
    key = (normalize_framework(framework), normalize_accelerator(accelerator))
    if key not in SUPPORTED_LOCAL_IMAGES:
        print(
            "Note: only CPU base image Dockerfiles were added locally for now. "
            f"The workflow can still reference {select_image(*key)}, but you must provide or publish that image."
        )


def imports_from_python(script: str) -> set[str]:
    try:
        tree = ast.parse(script)
    except SyntaxError:
        return set()

    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".", 1)[0])
    return imports


def normalize_package_name(name: str) -> str:
    cleaned = str(name).strip()
    if not cleaned:
        return ""
    cleaned = re.split(r"[<>=!~;\s\[]", cleaned, maxsplit=1)[0]
    return IMPORT_TO_PACKAGE.get(cleaned, cleaned)


def package_import_name(package_name: str) -> str:
    name = normalize_package_name(package_name)
    return PACKAGE_TO_IMPORT.get(name, name.replace("-", "_"))


def package_check_metadata(package_name: str) -> dict[str, Any]:
    name = normalize_package_name(package_name)
    return {
        "import": package_import_name(name),
        "attributes": PACKAGE_REQUIRED_ATTRIBUTES.get(name, []),
    }


def missing_runtime_packages(script: str, requested_packages: list[str]) -> list[str]:
    packages: set[str] = set()
    for package in requested_packages:
        normalized = normalize_package_name(package)
        if normalized:
            packages.add(normalized)

    for import_name in imports_from_python(script):
        if import_name in BASE_IMPORTS or import_name in sys.stdlib_module_names:
            continue
        packages.add(normalize_package_name(import_name))

    base_packages = {normalize_package_name(name) for name in BASE_IMPORTS}
    return sorted(package for package in packages if package and package not in base_packages)


def _filter_preinstalled_frameworks(packages: list[str]) -> list[str]:
    """Remove AMD-image frameworks and incompatible CUDA packages.

    These are pre-installed on AMD Quick Start Images and must not be
    overwritten so the host's ROCm / GPU integration stays intact.
    """
    allowed: list[str] = []
    for package in packages:
        normalized = normalize_package_name(package).lower()
        if normalized in CORE_DL_PACKAGES:
            continue
        if normalized in AMD_BLOCKED_PACKAGES:
            continue
        if normalized.startswith(AMD_PROTECTED_PACKAGE_PREFIXES):
            continue
        allowed.append(normalized)
    return allowed


def build_runtime_metadata(
    job_spec: dict[str, Any],
    script: str,
    requested_packages: list[str],
) -> dict[str, Any]:
    runtime = job_spec.setdefault("runtime", {})
    framework = normalize_framework(str(runtime.get("framework") or job_spec.get("framework") or "pytorch"))
    accelerator = normalize_accelerator(str(runtime.get("accelerator") or runtime.get("device") or "cpu"))
    image = str(runtime.get("image") or select_image(framework, accelerator))
    extra_packages = missing_runtime_packages(script, requested_packages)
    extra_packages = sorted(set(extra_packages) | set(_dataset_runtime_packages(job_spec.get("dataset") or {})))
    # Strip core DL frameworks — they are pre-installed on AMD Quick Start Images
    extra_packages = _filter_preinstalled_frameworks(extra_packages)
    artifact_contract = _artifact_contract(job_spec)

    runtime.update(
        {
            "framework": framework,
            "accelerator": accelerator,
            "image": image,
            "entrypoint": "generated/script.py",
            "outputs_dir": "/outputs",
            "workspace_dir": "/workspace",
            "extra_packages": extra_packages,
        }
    )

    return {
        "framework": framework,
        "accelerator": accelerator,
        "image": image,  # reference slug only — the runner executes natively, not via Docker
        "execution_mode": "native_host",  # workloads run on host Python, no container pulled
        "entrypoint": "generated/script.py",
        "outputs_dir": "/outputs",
        "extra_packages": extra_packages,
        "package_imports": {package: package_import_name(package) for package in extra_packages},
        "package_checks": {package: package_check_metadata(package) for package in extra_packages},
        **artifact_contract,
    }


def _artifact_contract(job_spec: dict[str, Any]) -> dict[str, list[str]]:
    task_type = str(job_spec.get("task_type") or job_spec.get("task") or "").lower()
    raw_expected = _normalize_expected_outputs(job_spec.get("expected_outputs"))

    # Strip redundant "outputs/" prefix from expected outputs — they are always
    # resolved relative to OUTPUTS, so "outputs/logs/runtime.log" really means
    # "logs/runtime.log".  Without this the "outputs" part becomes a required
    # directory under OUTPUTS, producing a double-nested MISS.
    expected_outputs: list[str] = []
    for item in raw_expected:
        stripped = item.removeprefix("outputs/").removeprefix("outputs\\").lstrip("/")
        if stripped and stripped != "outputs":
            expected_outputs.append(stripped)

    required_outputs: set[str] = {"environment.json"}
    required_dirs: set[str] = set()
    optional_outputs: set[str] = {"metrics.json", "training_logs.json"}
    optional_dirs: set[str] = {"logs", "plots"}

    if any(token in task_type for token in ("train", "finetune", "fine-tune")):
        required_outputs.update({"metrics.json", "training_logs.json"})
    elif any(token in task_type for token in ("eval", "test", "benchmark")):
        required_outputs.add("metrics.json")

    for output in expected_outputs:
        normalized = output.strip().lstrip("/")
        if not normalized:
            continue
        if normalized.endswith("/"):
            required_dirs.add(normalized.rstrip("/"))
            continue
        first_part = normalized.split("/", 1)[0]
        if "." in first_part:
            required_outputs.add(normalized)
        else:
            required_dirs.add(first_part)

    optional_outputs.difference_update(required_outputs)
    optional_dirs.difference_update(required_dirs)
    return {
        "required_outputs": sorted(required_outputs),
        "optional_outputs": sorted(optional_outputs),
        "required_dirs": sorted(required_dirs),
        "optional_dirs": sorted(optional_dirs),
    }


def _normalize_expected_outputs(raw_outputs: Any) -> list[str]:
    if not isinstance(raw_outputs, list):
        return []
    outputs: list[str] = []
    for item in raw_outputs:
        if isinstance(item, str):
            outputs.append(item)
        elif isinstance(item, dict):
            path = item.get("path") or item.get("name") or item.get("file") or item.get("dir")
            if path:
                outputs.append(str(path))
    return outputs


@lru_cache(maxsize=1)
def _load_outputs_helper_functions() -> set[str]:
    tree = ast.parse(_outputs_helper_source())
    return {
        node.name
        for node in ast.iter_child_nodes(tree)
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
    }


def _validate_outputs_imports(script: str) -> str:
    valid = _load_outputs_helper_functions()
    try:
        tree = ast.parse(script)
    except SyntaxError:
        return script

    fixes: list[tuple[int, str, str]] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ImportFrom) and node.module in ("outputs", "generated.outputs"):
            imported = [alias.name for alias in node.names]
            invalid = [name for name in imported if name not in valid]
            if not invalid:
                continue
            valid_names = [name for name in imported if name in valid]
            if not valid_names:
                new_line = f"# BROKEN: {ast.unparse(node)}  # none of these names exist in outputs helper"
            else:
                new_line = f"from outputs import {', '.join(valid_names)}"
            old_line = ast.unparse(node)
            fixes.append((node.lineno, old_line, new_line))

        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in ("outputs", "generated.outputs"):
                    continue
                if alias.name.startswith("outputs.") or alias.name.startswith("generated.outputs."):
                    used_name = alias.asname or alias.name.rsplit(".", 1)[-1]
                    if used_name not in valid:
                        line = ast.unparse(node)
                        fixes.append((node.lineno, line, f"# BROKEN: {line}  # '{used_name}' not in outputs helper"))

    if not fixes:
        return script

    lines = script.splitlines(keepends=True)
    for lineno, old, new in sorted(fixes, reverse=True):
        idx = lineno - 1
        if idx < len(lines) and old in lines[idx]:
            lines[idx] = lines[idx].replace(old, new, 1)
    result = "".join(lines)
    print(f"\n[validate] Fixed {len(fixes)} hallucinated import(s) in generated/script.py:\n", flush=True)
    for _, old, new in fixes:
        if new.startswith("# BROKEN"):
            print(f"  REMOVED: {old.strip()}", flush=True)
        else:
            print(f"  FIXED:   {old.strip()} -> {new.strip()}", flush=True)
    print(flush=True)
    return result


def _assert_valid_python_source(name: str, source: str) -> None:
    try:
        compile(source, name, "exec")
    except SyntaxError as exc:
        raise ValueError(f"Generated Python source has a syntax error in {name}: {exc}") from exc


def write_layered_payload(workspace: JobWorkspace, payload: dict[str, Any]) -> dict[str, Any]:
    job_spec = payload["job_spec"]
    job_spec["dataset"] = normalize_dataset_config(job_spec.get("dataset") or {})
    files = payload.get("files") or {}
    script = str(files.get("generated/script.py") or files.get("script.py") or "").strip()
    if not script:
        raise ValueError("Layered LLM response must include generated/script.py.")

    script = _validate_outputs_imports(script)
    outputs_source = _outputs_helper_source()
    dataset_manager_source = _dataset_manager_source()
    dataset_inspector_source = _dataset_inspector_source()
    installer_source = _runtime_installer_source()
    runner_source = _runtime_runner_source()
    validator_source = _runtime_validator_source()
    collector_source = _runtime_collector_source()
    bootstrap_source = _bootstrap_source()
    _assert_valid_python_source("generated/script.py", script)
    _assert_valid_python_source("generated/outputs.py", outputs_source)
    _assert_valid_python_source("dataset_manager.py", dataset_manager_source)
    _assert_valid_python_source("dataset_inspector.py", dataset_inspector_source)
    _assert_valid_python_source("runtime_installer.py", installer_source)
    _assert_valid_python_source("runtime_runner.py", runner_source)
    _assert_valid_python_source("runtime_validator.py", validator_source)
    _assert_valid_python_source("runtime_collector.py", collector_source)
    _assert_valid_python_source("runtime_bootstrap.py", bootstrap_source)

    requested_packages = (
        payload.get("extra_packages")
        or job_spec.get("extra_packages")
        or (job_spec.get("runtime") or {}).get("extra_packages")
        or []
    )
    if not isinstance(requested_packages, list):
        requested_packages = []

    metadata = build_runtime_metadata(job_spec, script, [str(item) for item in requested_packages])
    config = payload.get("config") or {
        "job_id": job_spec.get("job_id"),
        "objective": job_spec.get("objective"),
        "dataset": job_spec.get("dataset", {}),
        "runtime": job_spec.get("runtime", {}),
    }

    workspace.generated_dir.mkdir(parents=True, exist_ok=True)
    workspace.generated_dir.joinpath("script.py").write_text(script + "\n", encoding="utf-8")
    # Write outputs.py to the workspace root so that "from outputs import save_metrics"
    # resolves to the Python module (outputs.py) rather than being shadowed by the
    # OUTPUTS data directory (outputs/) that mkdir -p creates before the script runs.
    # A .py file on sys.path takes precedence over a namespace-package directory.
    workspace.payload_dir.joinpath("outputs.py").write_text(outputs_source, encoding="utf-8")
    workspace.payload_dir.joinpath("dataset_manager.py").write_text(dataset_manager_source, encoding="utf-8")
    workspace.payload_dir.joinpath("dataset_inspector.py").write_text(dataset_inspector_source, encoding="utf-8")
    workspace.payload_dir.joinpath("runtime_installer.py").write_text(installer_source, encoding="utf-8")
    workspace.payload_dir.joinpath("runtime_runner.py").write_text(runner_source, encoding="utf-8")
    workspace.payload_dir.joinpath("runtime_validator.py").write_text(validator_source, encoding="utf-8")
    workspace.payload_dir.joinpath("runtime_collector.py").write_text(collector_source, encoding="utf-8")
    workspace.payload_dir.joinpath("metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    workspace.payload_dir.joinpath("config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    workspace.payload_dir.joinpath("payload.json").write_text(
        json.dumps({"dataset": job_spec.get("dataset", {})}, indent=2),
        encoding="utf-8",
    )
    workspace.payload_dir.joinpath("datasets.json").write_text(
        json.dumps(job_spec.get("dataset", {}), indent=2),
        encoding="utf-8",
    )
    workspace.payload_dir.joinpath("Dockerfile").unlink(missing_ok=True)
    workspace.payload_dir.joinpath("requirements.txt").unlink(missing_ok=True)
    workspace.payload_dir.joinpath("runtime_bootstrap.py").write_text(bootstrap_source, encoding="utf-8")
    workspace.root.joinpath("job_spec.json").write_text(json.dumps(job_spec, indent=2), encoding="utf-8")
    return metadata


def _dataset_runtime_packages(dataset: dict[str, Any]) -> list[str]:
    dataset_type = str((dataset or {}).get("type") or "").lower()
    if dataset_type == "huggingface":
        return ["datasets"]
    return []


def write_dataset_inspection_payload(
    workspace: JobWorkspace,
    job_spec: dict[str, Any],
    framework: str,
    accelerator: str,
) -> dict[str, Any]:
    """Write the minimal payload used to download and inspect a dataset before generation."""
    job_spec["dataset"] = normalize_dataset_config(job_spec.get("dataset") or {})
    runtime = job_spec.setdefault("runtime", {})
    runtime["framework"] = normalize_framework(framework)
    runtime["accelerator"] = normalize_accelerator(accelerator)
    runtime["image"] = select_image(framework, accelerator)

    metadata = {
        "framework": runtime["framework"],
        "accelerator": runtime["accelerator"],
        "image": runtime["image"],
        "entrypoint": "dataset_inspector.py",
        "outputs_dir": "/outputs",
        "extra_packages": sorted(set(_dataset_runtime_packages(job_spec.get("dataset") or {}))),
        "package_imports": {
            package: package_import_name(package)
            for package in sorted(set(_dataset_runtime_packages(job_spec.get("dataset") or {})))
        },
        "package_checks": {
            package: package_check_metadata(package)
            for package in sorted(set(_dataset_runtime_packages(job_spec.get("dataset") or {})))
        },
    }

    workspace.payload_dir.mkdir(parents=True, exist_ok=True)
    workspace.generated_dir.mkdir(parents=True, exist_ok=True)
    workspace.payload_dir.joinpath("dataset_manager.py").write_text(_dataset_manager_source(), encoding="utf-8")
    workspace.payload_dir.joinpath("dataset_inspector.py").write_text(_dataset_inspector_source(), encoding="utf-8")
    workspace.payload_dir.joinpath("metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    workspace.payload_dir.joinpath("payload.json").write_text(
        json.dumps({"dataset": job_spec.get("dataset", {})}, indent=2),
        encoding="utf-8",
    )
    workspace.payload_dir.joinpath("datasets.json").write_text(
        json.dumps(job_spec.get("dataset", {}), indent=2),
        encoding="utf-8",
    )
    workspace.root.joinpath("job_spec.json").write_text(json.dumps(job_spec, indent=2), encoding="utf-8")
    return metadata


def _outputs_helper_source() -> str:
    return '''from __future__ import annotations

import json
import os
import platform
from pathlib import Path


OUTPUTS = Path(os.environ.get("DYNAMIC_CLOUD_OUTPUTS_DIR", "/outputs"))


def _ensure_outputs() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "logs").mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "plots").mkdir(parents=True, exist_ok=True)


def save_metrics(metrics: dict) -> Path:
    _ensure_outputs()
    path = OUTPUTS / "metrics.json"
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[outputs] Saved metrics.json: {metrics}", flush=True)
    return path


def log_epoch(epoch: int, metrics: dict, log_file: str = "training_logs.json") -> Path:
    _ensure_outputs()
    path = OUTPUTS / log_file
    logs = []
    if path.exists():
        logs = json.loads(path.read_text(encoding="utf-8"))
    logs.append({"epoch": epoch, **metrics})
    path.write_text(json.dumps(logs, indent=2), encoding="utf-8")
    return path


def save_training_history(history, file_name: str = "training_logs.json") -> Path:
    _ensure_outputs()
    path = OUTPUTS / file_name
    if hasattr(history, "history"):
        history = history.history
    if isinstance(history, dict):
        epochs = [dict({"epoch": i}, **{k: float(v[i]) for k, v in history.items()}) for i in range(len(next(iter(history.values()))))]
        path.write_text(json.dumps(epochs, indent=2), encoding="utf-8")
    else:
        path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"[outputs] Saved training history: {path}", flush=True)
    return path


def save_plot(figure, name: str) -> Path:
    _ensure_outputs()
    path = OUTPUTS / "plots" / name
    figure.savefig(str(path), bbox_inches="tight")
    print(f"[outputs] Saved plot: {path}", flush=True)
    return path


def save_model(model, path: str = "model", framework: str = "pytorch") -> Path:
    _ensure_outputs()
    ext = ".pth" if framework == "pytorch" else ".keras" if framework == "tensorflow" else ".joblib"
    model_path = OUTPUTS / f"{path}{ext}"
    if framework == "pytorch":
        import torch
        torch.save(model.state_dict(), model_path)
    elif framework == "tensorflow":
        model.save(str(model_path))
    else:
        import joblib
        joblib.dump(model, model_path)
    print(f"[outputs] Saved model: {model_path}", flush=True)
    return OUTPUTS  # return the outputs directory so any os.path.join(model_path, "model.pth") resolves correctly


def save_environment(extra: dict | None = None) -> Path:
    _ensure_outputs()
    env = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": {},
    }
    try:
        import importlib.metadata as md
        env["packages"] = {dist.name: dist.version for dist in md.distributions()}
    except Exception:
        pass
    if extra:
        env.update(extra)
    path = OUTPUTS / "environment.json"
    path.write_text(json.dumps(env, indent=2), encoding="utf-8")
    return path
'''


def _dataset_manager_source() -> str:
    return '''from __future__ import annotations

import json
import hashlib
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Any


WORKSPACE = Path(os.environ.get("DYNAMIC_CLOUD_WORKSPACE", "/workspace"))
DATASETS_ROOT = WORKSPACE / "prepared_datasets"
LOCAL_DATA = WORKSPACE / "data"


def load_payload(path: str | Path = WORKSPACE / "payload.json") -> dict[str, Any]:
    payload_path = Path(path)
    if not payload_path.exists():
        legacy = WORKSPACE / "datasets.json"
        if legacy.exists():
            return {"dataset": json.loads(legacy.read_text(encoding="utf-8"))}
        return {"dataset": {"type": "none"}}
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("payload.json must contain a JSON object.")
    return payload


def prepare_dataset(config: dict[str, Any] | None = None) -> Path | None:
    config = normalize_dataset_config(config or load_payload().get("dataset") or {})
    dataset_type = str(config.get("type") or "none").lower()
    DATASETS_ROOT.mkdir(parents=True, exist_ok=True)

    if dataset_type == "none":
        print("[dataset] No external dataset source was configured.", flush=True)
        return None
    if dataset_type == "local":
        path = Path(str(config.get("container_data_dir") or LOCAL_DATA))
        if not path.exists():
            raise FileNotFoundError(f"Local dataset directory was not uploaded: {path}")
        return _write_info(config, path)
    if dataset_type == "huggingface":
        return _prepare_huggingface(config)
    if dataset_type == "github":
        return _prepare_github(config)
    raise ValueError(f"Unsupported dataset.type: {dataset_type}")


''' + _dataset_normalizer_source() + '''


def _prepare_huggingface(config: dict[str, Any]) -> Path:
    dataset_id = str(config.get("id") or config.get("repo") or "").strip()
    if not dataset_id:
        raise ValueError("Hugging Face dataset requires dataset.id or dataset.repo.")

    revision = config.get("revision")
    if revision is None and "@" in dataset_id:
        dataset_id, revision = dataset_id.split("@", 1)

    target = _dataset_target(config, dataset_id)
    if (target / "dataset_info.json").exists() or (target / "state.json").exists():
        print(f"[dataset] Reusing Hugging Face dataset at {target}", flush=True)
        return _write_info(config, target)

    kwargs: dict[str, Any] = {"cache_dir": str(DATASETS_ROOT / ".hf_cache")}
    if revision:
        kwargs["revision"] = revision
    config_name = config.get("config_name") or config.get("subset") or config.get("dataset_config")
    if config_name:
        kwargs["name"] = config_name

    try:
        _load_hf_dataset(dataset_id, target, config.get("split"), kwargs)
    except _HfNamespaceError:
        _retry_hf_dataset_with_compat_libs(dataset_id, target, config.get("split"), kwargs)
    return _write_info(config, target)


def _load_hf_dataset(
    dataset_id: str,
    target: Path,
    split: str | None,
    kwargs: dict[str, Any],
) -> None:
    import datasets as hf_datasets

    load_dataset = getattr(hf_datasets, "load_dataset", None)
    if not callable(load_dataset):
        raise RuntimeError(
            "The installed 'datasets' package does not expose load_dataset. "
            "Install a compatible Hugging Face datasets release."
        )

    try:
        if split:
            dataset = load_dataset(dataset_id, split=split, **kwargs)
        else:
            dataset = load_dataset(dataset_id, **kwargs)
    except Exception as exc:
        if "namespace/name" in str(exc) or "HfUriError" in type(exc).__name__:
            raise _HfNamespaceError(dataset_id) from exc
        raise

    target.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(target))


class _HfNamespaceError(ValueError):
    def __init__(self, dataset_id: str) -> None:
        super().__init__(
            f"HuggingFace dataset '{dataset_id}' lacks a namespace. "
            f"The installed huggingface_hub version requires 'namespace/name' format. "
            f"See https://huggingface.co/docs/hub/datasets-downloading"
        )


def _retry_hf_dataset_with_compat_libs(
    dataset_id: str,
    target: Path,
    split: str | None,
    kwargs: dict[str, Any],
) -> None:
    import json
    import subprocess
    import sys
    from pathlib import Path

    print(
        f"[dataset] Retrying dataset '{dataset_id}' with compatible "
        f"huggingface_hub version...",
        flush=True,
    )
    target_str = str(target)
    kwargs_json = json.dumps(kwargs)

    # Cache compat libs in a stable directory under the dataset cache root
    # so they are reused across retries — no repeated pip installs.
    old_libs_dir = str(DATASETS_ROOT / ".hf_compat_overlay")
    hf_hub_dir = os.path.join(old_libs_dir, "huggingface_hub")
    datasets_dir = os.path.join(old_libs_dir, "datasets")

    if not (os.path.isdir(hf_hub_dir) and os.path.isdir(datasets_dir)):
        print(f"[dataset] Installing compat libs to {old_libs_dir}...", flush=True)
        os.makedirs(old_libs_dir, exist_ok=True)
        subprocess.check_call(
            [
                sys.executable, "-m", "pip", "install",
                "--target", old_libs_dir,
                "--ignore-installed",
                "--quiet",
                "huggingface_hub<0.24.0", "datasets<3.0.0",
            ],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    else:
        print(f"[dataset] Reusing cached compat libs at {old_libs_dir}", flush=True)

    lines = [
        "import sys",
        f"sys.path.insert(0, {old_libs_dir!r})",
        "import datasets as hf_datasets",
        "load_dataset = getattr(hf_datasets, 'load_dataset', None)",
        "if not callable(load_dataset): raise RuntimeError('datasets package is missing load_dataset')",
        "from pathlib import Path",
    ]
    if split:
        lines.append(f"ds = load_dataset({dataset_id!r}, split={split!r}, **{kwargs_json})")
    else:
        lines.append(f"ds = load_dataset({dataset_id!r}, **{kwargs_json})")
    lines.append(f"Path({target_str!r}).mkdir(parents=True, exist_ok=True)")
    lines.append(f"ds.save_to_disk({target_str!r})")
    lines.append('print("_HF_RETRY_OK")')

    code = "\\n".join(lines)
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=600,
    )
    if "_HF_RETRY_OK" not in result.stdout:
        stderr = result.stderr.strip()
        raise RuntimeError(
            f"Failed to load HuggingFace dataset '{dataset_id}' "
            f"even after downgrading huggingface_hub. "
            f"Try specifying the full dataset path with a "
            f"namespace prefix (e.g., 'namespace/{dataset_id}')."
            + (f" Details: {stderr}" if stderr else "")
        )


def _prepare_github(config: dict[str, Any]) -> Path:
    url = str(config.get("url") or "").strip()
    if not url:
        raise ValueError("GitHub dataset requires dataset.url.")

    target = _dataset_target(config, url)
    if target.exists() and any(target.iterdir()):
        return _write_info(config, target)
    # Requested branch may be None — in that case use the repo's *default*
    # branch (do NOT force "main"; many repos default to "master" or another
    # name and `git clone --branch main` would fail).
    requested_branch = config.get("branch")

    # Probe for git; install if missing; fall back to tarball download.
    has_git = False
    try:
        subprocess.check_call(["which", "git"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        has_git = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    if not has_git:
        print("[dataset] git not found on VM; attempting to install...", flush=True)
        try:
            subprocess.check_call(
                ["apt-get", "update", "-qq"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120,
            )
            subprocess.check_call(
                ["apt-get", "install", "-y", "-qq", "git"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120,
            )
            has_git = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

    if has_git:
        command = ["git", "clone", "--depth", "1"]
        if requested_branch:
            command.extend(["--branch", str(requested_branch)])
        command.extend([url, str(target)])
        subprocess.check_call(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return _write_info(config, target)

    # Fallback: download as tarball via curl.  The GitHub archive URL needs an
    # explicit branch, so when the user did not name one we try the common
    # defaults (main, then master) until one resolves.
    print(f"[dataset] Installing git failed; downloading {url} as tarball...", flush=True)
    candidate_branches = [str(requested_branch)] if requested_branch else ["main", "master"]
    target.mkdir(parents=True, exist_ok=True)
    tarball_path = target / "source.tar.gz"
    last_exc: Exception | None = None
    for branch in candidate_branches:
        tarball_url = f"{url.rstrip('/')}/archive/refs/heads/{branch}.tar.gz"
        try:
            subprocess.check_call(
                ["curl", "-fsSL", "-o", str(tarball_path), tarball_url],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300,
            )
            subprocess.check_call(
                ["tar", "-xzf", str(tarball_path), "-C", str(target), "--strip-components=1"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120,
            )
            tarball_path.unlink(missing_ok=True)
            return _write_info(config, target)
        except subprocess.CalledProcessError as exc:
            last_exc = exc
            tarball_path.unlink(missing_ok=True)
            print(f"[dataset] tarball for branch '{branch}' failed; trying next...", flush=True)
            continue
    raise RuntimeError(
        f"Could not clone or download GitHub dataset from {url}. "
        f"Both git clone and tarball download failed (tried branches: "
        f"{', '.join(candidate_branches)}). "
        f"Verify the URL and network access on the VM."
    ) from last_exc


def _write_info(config: dict[str, Any], dataset_path: Path) -> Path:
    dataset_path = dataset_path.resolve()
    info = {
        "dataset_path": str(dataset_path),
        "dataset": config,
    }
    info_path = WORKSPACE / "dataset_info.json"
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    os.environ["DATASET_PATH"] = str(dataset_path)
    os.environ["DATASET_INFO_PATH"] = str(info_path)
    return dataset_path


def _replace_with_copy(source: Path, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    if source.is_dir():
        shutil.copytree(source, target)
    else:
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target / source.name)


def _dataset_target(config: dict[str, Any], display_name: str) -> Path:
    name = _safe_name(display_name)
    return DATASETS_ROOT / f"{name}-{_dataset_hash(config)}"


def _dataset_hash(config: dict[str, Any]) -> str:
    ignored = {
        "container_data_dir",
        "description",
        "local_paths",
        "path",
    }
    stable = {
        key: value
        for key, value in normalize_dataset_config(config).items()
        if key not in ignored and value not in (None, "", [], {})
    }
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]


def _safe_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "._-" else "-" for char in value.strip())
    cleaned = cleaned.strip(".-")
    return cleaned[:120] or "dataset"


if __name__ == "__main__":
    prepare_dataset()
'''


def _dataset_normalizer_source() -> str:
    resolver = inspect.getsource(resolve_hf_dataset_id)
    normalizer = inspect.getsource(normalize_dataset_config)
    return resolver + "\n\n\n" + normalizer


def _dataset_inspector_source() -> str:
    return '''from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from dataset_manager import load_payload, prepare_dataset


WORKSPACE = Path(os.environ.get("DYNAMIC_CLOUD_WORKSPACE", "/workspace"))
OUTPUTS = Path(os.environ.get("DYNAMIC_CLOUD_OUTPUTS_DIR", "/outputs"))
MAX_TREE_DEPTH = 5
MAX_DIR_ENTRIES = 80
MAX_SAMPLE_FILES = 80


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    payload = load_payload(WORKSPACE / "payload.json")
    dataset_config = payload.get("dataset") or {}
    dataset_path = prepare_dataset(dataset_config)
    metadata = inspect_dataset(Path(dataset_path).resolve() if dataset_path else None, dataset_config)
    write_metadata(metadata)


def inspect_dataset(dataset_path: Path | None, dataset_config: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "dataset": dataset_config,
        "dataset_path": str(dataset_path) if dataset_path else None,
        "exists": bool(dataset_path and dataset_path.exists()),
        "structure": None,
        "summary": {},
        "detected_layout": {},
        "samples": {},
    }
    if not dataset_path or not dataset_path.exists():
        metadata["summary"] = {"message": "No external dataset was prepared."}
        return metadata

    files = [path for path in dataset_path.rglob("*") if path.is_file()]
    dirs = [path for path in dataset_path.rglob("*") if path.is_dir()]
    extension_counts = Counter(path.suffix.lower() or "<no_ext>" for path in files)
    top_level = sorted(path.name for path in dataset_path.iterdir())

    metadata["structure"] = build_tree(dataset_path)
    metadata["summary"] = {
        "root_name": dataset_path.name,
        "top_level_entries": top_level[:MAX_DIR_ENTRIES],
        "total_files": len(files),
        "total_directories": len(dirs),
        "extension_counts": dict(extension_counts.most_common(40)),
        "sample_files": [str(path.relative_to(dataset_path)) for path in files[:MAX_SAMPLE_FILES]],
    }
    metadata["detected_layout"] = detect_layout(dataset_path)
    metadata["samples"] = inspect_known_formats(dataset_path)
    return metadata


def build_tree(root: Path, depth: int = 0) -> dict[str, Any]:
    node: dict[str, Any] = {
        "name": root.name,
        "type": "directory" if root.is_dir() else "file",
    }
    if root.is_file():
        node["size_bytes"] = root.stat().st_size
        return node
    if depth >= MAX_TREE_DEPTH:
        node["truncated"] = True
        node["child_count"] = len(list(root.iterdir()))
        return node

    children = sorted(root.iterdir(), key=lambda path: (path.is_file(), path.name.lower()))
    visible = children[:MAX_DIR_ENTRIES]
    node["children"] = [build_tree(child, depth + 1) for child in visible]
    if len(children) > len(visible):
        node["truncated"] = True
        node["omitted_children"] = len(children) - len(visible)
    return node


def detect_layout(root: Path) -> dict[str, Any]:
    split_names = {"train", "training", "val", "valid", "validation", "test", "testing"}
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
    result: dict[str, Any] = {
        "kind": "unknown",
        "splits": {},
        "class_directories": [],
        "label_files": [],
    }

    label_files = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name.lower() in {"labels.csv", "train.csv", "test.csv", "metadata.csv", "annotations.csv"}
    ]
    result["label_files"] = label_files[:40]

    for child in sorted(path for path in root.iterdir() if path.is_dir()):
        lowered = child.name.lower()
        if lowered in split_names:
            class_dirs = [item.name for item in sorted(child.iterdir()) if item.is_dir()]
            image_count = sum(1 for file_path in child.rglob("*") if file_path.is_file() and file_path.suffix.lower() in image_exts)
            result["splits"][child.name] = {
                "class_directories": class_dirs[:80],
                "class_count": len(class_dirs),
                "image_file_count": image_count,
            }

    top_class_dirs = [item.name for item in sorted(root.iterdir()) if item.is_dir() and item.name.lower() not in split_names]
    if result["splits"]:
        result["kind"] = "split_directory_tree"
    elif top_class_dirs and any((root / name).is_dir() for name in top_class_dirs):
        image_counts = {
            name: sum(1 for file_path in (root / name).rglob("*") if file_path.is_file() and file_path.suffix.lower() in image_exts)
            for name in top_class_dirs[:80]
        }
        if any(image_counts.values()):
            result["kind"] = "class_directory_tree"
            result["class_directories"] = [{"name": name, "image_file_count": image_counts[name]} for name in top_class_dirs[:80]]
    elif label_files:
        result["kind"] = "tabular_or_annotation_files"

    return result


def inspect_known_formats(root: Path) -> dict[str, Any]:
    samples: dict[str, Any] = {}
    csv_files = [path for path in root.rglob("*.csv") if path.is_file()]
    if csv_files:
        samples["csv_files"] = [sample_text(path) for path in csv_files[:5]]

    json_files = [path for path in root.rglob("*.json") if path.is_file()]
    if json_files:
        samples["json_files"] = [sample_text(path) for path in json_files[:5]]

    try:
        from datasets import load_from_disk
        loaded = load_from_disk(str(root))
        samples["huggingface"] = describe_huggingface_dataset(loaded)
    except Exception:
        pass
    return samples


def describe_huggingface_dataset(dataset: Any) -> dict[str, Any]:
    description: dict[str, Any] = {
        "type": type(dataset).__name__,
    }
    if hasattr(dataset, "keys"):
        description["splits"] = {
            key: {
                "num_rows": len(value),
                "features": list(getattr(value, "features", {}).keys()),
                "feature_types": {name: str(feature) for name, feature in getattr(value, "features", {}).items()},
            }
            for key, value in dataset.items()
        }
    else:
        description["num_rows"] = len(dataset)
        description["features"] = list(getattr(dataset, "features", {}).keys())
        description["feature_types"] = {name: str(feature) for name, feature in getattr(dataset, "features", {}).items()}
    return description


def sample_text(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return {"path": str(path), "error": str(exc)}
    lines = text.splitlines()
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "first_lines": lines[:5],
    }


def write_metadata(metadata: dict[str, Any]) -> None:
    for path in (WORKSPACE / "dataset_metadata.json", OUTPUTS / "dataset_metadata.json"):
        path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    summary = metadata.get("summary", {})
    layout = metadata.get("detected_layout", {})
    print(f"[dataset] Wrote dataset_metadata.json — {summary.get('total_files', 0)} files, "
          f"{summary.get('total_directories', 0)} dirs, "
          f"layout: {layout.get('kind', 'unknown')}", flush=True)


if __name__ == "__main__":
    main()
'''


def _runtime_installer_source() -> str:
    return '''from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


WORKSPACE = Path(os.environ.get("DYNAMIC_CLOUD_WORKSPACE", "/workspace"))
PACKAGE_OVERLAY = Path(os.environ.get("DYNAMIC_CLOUD_PACKAGE_OVERLAY", "/opt/dynamic-cloud/python-packages"))
DEFAULT_REQUIRED_ATTRIBUTES = {
    "azure-storage-blob": ["BlobServiceClient"],
    "azure.storage.blob": ["BlobServiceClient"],
    "datasets": ["load_dataset", "load_from_disk"],
}
PROTECTED_PACKAGE_PREFIXES = ("nvidia-", "cuda", "cupy-cuda")
PROTECTED_PACKAGES = {"torch", "torchvision", "torchaudio", "pytorch", "tensorflow", "keras", "tf", "jax", "jaxlib", "rocm", "amd"}
BLOCKED_PACKAGES = {"nvidia-ml-py3", "pycuda", "torch-cuda", "tf-nightly", "tensorflow-cpu", "tensorflow-gpu"}


''' + _package_install_helper_source() + '''


def main() -> Path | None:
    _activate_overlay()
    metadata = json.loads((WORKSPACE / "metadata.json").read_text(encoding="utf-8"))
    packages = [
        str(package)
        for package in (metadata.get("extra_packages") or [])
        if _package_is_safe_for_amd(str(package))
    ]
    package_imports = dict(metadata.get("package_imports") or {})
    package_checks = dict(metadata.get("package_checks") or {})
    missing = [
        package
        for package in packages
        if not _package_is_available(package, package_imports.get(package, package), package_checks.get(package))
    ]

    if missing:
        print(f"Installing missing runtime packages ({len(missing)}): {', '.join(missing)}", flush=True)
        # Do not touch the Python environment baked into the AMD image.
        # In particular, Debian-owned packages have no pip RECORD and cannot
        # be uninstalled safely.  A per-container overlay keeps ROCm and the
        # preinstalled framework intact while still allowing job dependencies.
        PACKAGE_OVERLAY.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "--target", str(PACKAGE_OVERLAY), *missing],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        print(f"✓ Installed {len(missing)} package(s)", flush=True)
        _activate_overlay()
    else:
        print("All runtime packages already available.", flush=True)

    return _prepare_dataset()


def _package_is_safe_for_amd(package: str) -> bool:
    normalized = package.strip().lower().replace("_", "-")
    return bool(normalized) and normalized not in PROTECTED_PACKAGES and normalized not in BLOCKED_PACKAGES and not normalized.startswith(PROTECTED_PACKAGE_PREFIXES)


def _activate_overlay() -> None:
    overlay = str(PACKAGE_OVERLAY)
    if overlay not in sys.path:
        sys.path.insert(0, overlay)


def _prepare_dataset() -> Path | None:
    manager_path = WORKSPACE / "dataset_manager.py"
    payload_path = WORKSPACE / "payload.json"
    if not manager_path.exists():
        print("[dataset] No dataset manager found; skipping dataset preparation.", flush=True)
        return None
    if str(WORKSPACE) not in sys.path:
        sys.path.insert(0, str(WORKSPACE))
    from dataset_manager import load_payload, prepare_dataset

    payload = load_payload(payload_path)
    dataset_config = payload.get("dataset") or {}
    dataset_path = prepare_dataset(dataset_config)
    return Path(dataset_path) if dataset_path else None


if __name__ == "__main__":
    main()
'''


def _package_install_helper_source() -> str:
    return '''import importlib
import importlib.util


def _package_is_available(package: str, import_name: str, check: dict | None = None) -> bool:
    check = check or {}
    import_name = str(check.get("import") or import_name)
    required_attributes = list(
        check.get("attributes")
        or DEFAULT_REQUIRED_ATTRIBUTES.get(package)
        or DEFAULT_REQUIRED_ATTRIBUTES.get(import_name)
        or []
    )
    try:
        if importlib.util.find_spec(import_name) is None:
            return False
        module = importlib.import_module(import_name)
    except ModuleNotFoundError:
        return False
    except ImportError:
        return False
    return all(hasattr(module, attr) for attr in required_attributes)
'''


def dataset_package_install_code() -> str:
    required_attributes = {
        **PACKAGE_REQUIRED_ATTRIBUTES,
        "azure.storage.blob": PACKAGE_REQUIRED_ATTRIBUTES.get("azure-storage-blob", []),
    }
    source = '''from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_REQUIRED_ATTRIBUTES = __REQUIRED_ATTRIBUTES__
PACKAGE_OVERLAY = Path(os.environ.get("DYNAMIC_CLOUD_PACKAGE_OVERLAY", "/opt/dynamic-cloud/python-packages"))
PROTECTED_PACKAGE_PREFIXES = ("nvidia-", "cuda", "cupy-cuda")
PROTECTED_PACKAGES = {"torch", "torchvision", "torchaudio", "pytorch", "tensorflow", "keras", "tf", "jax", "jaxlib", "rocm", "amd"}
BLOCKED_PACKAGES = {"nvidia-ml-py3", "pycuda", "torch-cuda", "tf-nightly", "tensorflow-cpu", "tensorflow-gpu"}


''' + _package_install_helper_source() + '''


def _package_is_safe_for_amd(package: str) -> bool:
    normalized = package.strip().lower().replace("_", "-")
    return bool(normalized) and normalized not in PROTECTED_PACKAGES and normalized not in BLOCKED_PACKAGES and not normalized.startswith(PROTECTED_PACKAGE_PREFIXES)


WORKSPACE_ROOT = Path(os.environ.get("DYNAMIC_CLOUD_WORKSPACE", "/workspace"))
metadata = json.load(open(str(WORKSPACE_ROOT / "metadata.json")))
package_imports = metadata.get("package_imports") or {}
package_checks = metadata.get("package_checks") or {}
missing = [
    package
    for package in (metadata.get("extra_packages") or [])
    if _package_is_safe_for_amd(str(package))
    and not _package_is_available(package, package_imports.get(package, package), package_checks.get(package))
]
print("Missing dataset packages:", ", ".join(missing) if missing else "none", flush=True)
if missing:
    # Install into an overlay rather than modifying Debian/ROCm packages in
    # the prebuilt AMD image.  --target never performs an uninstall.
    PACKAGE_OVERLAY.mkdir(parents=True, exist_ok=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--target", str(PACKAGE_OVERLAY), *missing],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
'''
    return source.replace("__REQUIRED_ATTRIBUTES__", repr(required_attributes))


def _runtime_runner_source() -> str:
    return '''from __future__ import annotations

import json
import os
import sys
from pathlib import Path


WORKSPACE = Path(os.environ.get("DYNAMIC_CLOUD_WORKSPACE", "/workspace"))
OUTPUTS = Path(os.environ.get("DYNAMIC_CLOUD_OUTPUTS_DIR", "/outputs"))

# Remote package overlay -- must be on sys.path for imports to resolve.
_PACKAGE_OVERLAY = os.environ.get(
    "DYNAMIC_CLOUD_PACKAGE_OVERLAY", "/opt/dynamic-cloud/python-packages"
)
if os.path.isdir(_PACKAGE_OVERLAY) and _PACKAGE_OVERLAY not in sys.path:
    sys.path.insert(0, _PACKAGE_OVERLAY)


def main() -> None:
    metadata = json.loads((WORKSPACE / "metadata.json").read_text(encoding="utf-8"))
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    script = WORKSPACE / metadata.get("entrypoint", "generated/script.py")
    if not script.exists():
        raise SystemExit(f"Entrypoint not found: {script}")

    # Set environment so the child script can find dataset paths.
    dataset_path = _load_dataset_path()
    if dataset_path:
        os.environ["DATASET_PATH"] = str(dataset_path)
        os.environ["DATASET_INFO_PATH"] = str(WORKSPACE / "dataset_info.json")
    if (WORKSPACE / "dataset_metadata.json").exists():
        os.environ["DATASET_METADATA_PATH"] = str(WORKSPACE / "dataset_metadata.json")

    # Ensure the workspace is on sys.path so intra-workspace imports resolve.
    ws_str = str(WORKSPACE)
    if ws_str not in sys.path:
        sys.path.insert(0, ws_str)

    # Execute the generated script in the *current* process so it inherits
    # every env var, sys.path entry, and installed module without risk of
    # env-propagation bugs across a subprocess boundary.
    code = script.read_text(encoding="utf-8")
    compiled = compile(code, str(script), "exec", dont_inherit=False)
    globals_ = {
        "__name__": "__main__",
        "__file__": str(script),
        "__builtins__": __builtins__,
    }
    sys.stdout.flush()
    exec(compiled, globals_)


def _load_dataset_path() -> Path | None:
    info_path = WORKSPACE / "dataset_info.json"
    if not info_path.exists():
        return None
    info = json.loads(info_path.read_text(encoding="utf-8"))
    dataset_path = info.get("dataset_path")
    return Path(dataset_path) if dataset_path else None


if __name__ == "__main__":
    main()
'''


def _runtime_validator_source() -> str:
    return '''from __future__ import annotations

import json
import os
from pathlib import Path


WORKSPACE = Path(os.environ.get("DYNAMIC_CLOUD_WORKSPACE", "/workspace"))
OUTPUTS = Path(os.environ.get("DYNAMIC_CLOUD_OUTPUTS_DIR", "/outputs"))


def main() -> None:
    metadata = _load_metadata()
    required_outputs = _as_list(metadata.get("required_outputs")) or ["environment.json"]
    optional_outputs = _as_list(metadata.get("optional_outputs"))
    required_dirs = _as_list(metadata.get("required_dirs"))
    optional_dirs = _as_list(metadata.get("optional_dirs"))

    (OUTPUTS / "logs").mkdir(parents=True, exist_ok=True)
    report_lines = []
    print("\\n" + "=" * 60, flush=True)
    print("OUTPUT REPORT", flush=True)
    print("=" * 60, flush=True)

    all_ok = True
    for name in required_outputs:
        ok = _report_file(name, required=True, report_lines=report_lines)
        all_ok = all_ok and ok
    for name in optional_outputs:
        _report_file(name, required=False, report_lines=report_lines)

    for name in required_dirs:
        ok = _report_dir(name, required=True, report_lines=report_lines)
        all_ok = all_ok and ok
    for name in optional_dirs:
        _report_dir(name, required=False, report_lines=report_lines)

    print("-" * 60, flush=True)
    print(f"Outputs root: {OUTPUTS}", flush=True)
    print("=" * 60, flush=True)
    (OUTPUTS / "logs" / "output_report.txt").write_text("\\n".join(report_lines) + "\\n", encoding="utf-8")
    if not all_ok:
        # Debug: list everything actually in OUTPUTS so the mismatch is obvious.
        actual = sorted(path.relative_to(OUTPUTS).as_posix()
                        for path in OUTPUTS.rglob("*") if path.is_file())
        if actual:
            print("\\n[debug] Files actually in OUTPUTS:", flush=True)
            for f in actual:
                print(f"  {f}", flush=True)
        raise SystemExit("One or more required output artifacts are missing.")


def _load_metadata() -> dict:
    path = WORKSPACE / "metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _as_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip().strip("/") for item in value if str(item).strip().strip("/")]


def _report_file(name: str, required: bool, report_lines: list[str]) -> bool:
    path = OUTPUTS / name
    if path.exists() and path.is_file():
        message = f"  [OK]   {name} ({_fmt_size(path.stat().st_size)})"
        print(message, flush=True)
        report_lines.append(message)
        return True
    label = "MISS" if required else "WARN"
    suffix = "" if required else " (optional)"
    message = f"  [{label}] {name}{suffix}"
    print(message, flush=True)
    report_lines.append(message)
    return not required


def _report_dir(name: str, required: bool, report_lines: list[str]) -> bool:
    path = OUTPUTS / name
    if path.exists() and path.is_dir():
        count = len(list(path.iterdir()))
        message = f"  [OK]   {name}/ ({count} files)"
        print(message, flush=True)
        report_lines.append(message)
        return True
    label = "MISS" if required else "WARN"
    suffix = "" if required else " (optional)"
    message = f"  [{label}] {name}/{suffix}"
    print(message, flush=True)
    report_lines.append(message)
    return not required


def _fmt_size(size: int) -> str:
    if size < 1024:
        return f"{size}B"
    elif size < 1024 ** 2:
        return f"{size / 1024:.1f}KB"
    return f"{size / 1024 ** 2:.1f}MB"


if __name__ == "__main__":
    main()
'''


def _runtime_collector_source() -> str:
    return '''from __future__ import annotations

import json
import os
from pathlib import Path


OUTPUTS = Path(os.environ.get("DYNAMIC_CLOUD_OUTPUTS_DIR", "/outputs"))


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for path in sorted(OUTPUTS.rglob("*")):
        if path.is_file() and path.name != "run_manifest.json":
            artifacts.append(
                {
                    "path": path.relative_to(OUTPUTS).as_posix(),
                    "size_bytes": path.stat().st_size,
                }
            )
    manifest_path = OUTPUTS / "run_manifest.json"
    manifest_path.write_text(json.dumps({"artifacts": artifacts}, indent=2), encoding="utf-8")
    print(f"[collector] Wrote run_manifest.json with {len(artifacts)} artifact(s).", flush=True)


if __name__ == "__main__":
    main()
'''


def _bootstrap_source() -> str:
    return '''from __future__ import annotations

import runtime_collector
import runtime_installer
import runtime_runner
import runtime_validator


def main() -> None:
    runtime_installer.main()
    runtime_runner.main()
    runtime_validator.main()
    runtime_collector.main()


if __name__ == "__main__":
    main()
'''
