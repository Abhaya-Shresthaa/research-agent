from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dynamic_cloud.config import RUNTIME_WORKSPACE
from dynamic_cloud.dataset_config import normalize_dataset_config


@dataclass(frozen=True)
class JobWorkspace:
    job_id: str
    root: Path
    payload_dir: Path
    generated_dir: Path
    data_dir: Path


def load_job_spec(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        spec = json.load(handle)
    if not isinstance(spec, dict):
        raise ValueError("Job spec must be a JSON object.")
    if not spec.get("job_id"):
        spec["job_id"] = "dynamic-job"
    return spec


def safe_job_id(raw: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "-", raw.strip()).strip(".-")
    return value or "dynamic-job"


def prepare_workspace(spec: dict[str, Any], reset: bool = False) -> JobWorkspace:
    job_id = safe_job_id(str(spec["job_id"]))
    spec["dataset"] = normalize_dataset_config(spec.get("dataset") or {})
    root = RUNTIME_WORKSPACE / job_id
    payload_dir = root / "payload"
    generated_dir = payload_dir / "generated"
    data_dir = payload_dir / "data"

    if reset and root.exists():
        shutil.rmtree(root)

    generated_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    (root / "job_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")

    prepare_dataset(spec.get("dataset") or {}, data_dir)
    return JobWorkspace(job_id=job_id, root=root, payload_dir=payload_dir, generated_dir=generated_dir, data_dir=data_dir)


def prepare_dataset(dataset: dict[str, Any], data_dir: Path) -> None:
    config = normalize_dataset_config(dataset)
    if config.get("type") == "local":
        copy_dataset_paths(config, data_dir)


def copy_dataset_paths(dataset: dict[str, Any], data_dir: Path) -> None:
    local_paths = dataset.get("local_paths") or []
    for raw_path in local_paths:
        source = Path(raw_path).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {source}")
        target = data_dir / source.name
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)


def validate_payload(workspace: JobWorkspace) -> None:
    script_path = workspace.generated_dir / "script.py"
    metadata_path = workspace.payload_dir / "metadata.json"
    dockerfile_path = workspace.payload_dir / "Dockerfile"
    requirements_path = workspace.payload_dir / "requirements.txt"

    if metadata_path.exists():
        _refresh_layered_helpers(workspace)
        required_files = [
            metadata_path,
            workspace.payload_dir / "runtime_bootstrap.py",
            workspace.payload_dir / "runtime_installer.py",
            workspace.payload_dir / "runtime_runner.py",
            workspace.payload_dir / "runtime_validator.py",
            workspace.payload_dir / "runtime_collector.py",
            workspace.payload_dir / "dataset_manager.py",
            workspace.payload_dir / "dataset_inspector.py",
            workspace.payload_dir / "payload.json",
            script_path,
            workspace.generated_dir / "outputs.py",
        ]
    else:
        required_files = [
            dockerfile_path,
            requirements_path,
            script_path,
        ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(f"Required payload file missing: {file_path}")
        text = file_path.read_text(encoding="utf-8").strip()
        if not text and file_path.name != "requirements.txt":
            raise ValueError(f"Required payload file is empty: {file_path}")
        if "placeholder" in text.lower() and file_path.name != "requirements.txt":
            raise ValueError(f"Payload still contains placeholder content: {file_path}")
        if file_path.suffix == ".py":
            try:
                compile(text, str(file_path), "exec")
            except SyntaxError as exc:
                raise ValueError(f"Payload Python syntax error in {file_path}: {exc}") from exc

    if metadata_path.exists():
        stale_files = [path for path in (dockerfile_path, requirements_path) if path.exists()]
        if stale_files:
            names = ", ".join(path.name for path in stale_files)
            raise ValueError(f"Layered payload contains stale legacy file(s): {names}")


def _refresh_layered_helpers(workspace: JobWorkspace) -> None:
    from dynamic_cloud.runtime_layers import (
        _bootstrap_source,
        _dataset_inspector_source,
        _dataset_manager_source,
        _outputs_helper_source,
        _runtime_collector_source,
        _runtime_installer_source,
        _runtime_runner_source,
        _runtime_validator_source,
    )

    workspace.payload_dir.mkdir(parents=True, exist_ok=True)
    workspace.generated_dir.mkdir(parents=True, exist_ok=True)
    workspace.payload_dir.joinpath("dataset_manager.py").write_text(_dataset_manager_source(), encoding="utf-8")
    workspace.payload_dir.joinpath("dataset_inspector.py").write_text(_dataset_inspector_source(), encoding="utf-8")
    workspace.payload_dir.joinpath("runtime_installer.py").write_text(_runtime_installer_source(), encoding="utf-8")
    workspace.payload_dir.joinpath("runtime_runner.py").write_text(_runtime_runner_source(), encoding="utf-8")
    workspace.payload_dir.joinpath("runtime_validator.py").write_text(_runtime_validator_source(), encoding="utf-8")
    workspace.payload_dir.joinpath("runtime_collector.py").write_text(_runtime_collector_source(), encoding="utf-8")
    workspace.payload_dir.joinpath("runtime_bootstrap.py").write_text(_bootstrap_source(), encoding="utf-8")
    workspace.generated_dir.joinpath("outputs.py").write_text(_outputs_helper_source(), encoding="utf-8")
    payload_json = workspace.payload_dir / "payload.json"
    if not payload_json.exists():
        job_spec_path = workspace.root / "job_spec.json"
        dataset = {"type": "none"}
        if job_spec_path.exists():
            spec = json.loads(job_spec_path.read_text(encoding="utf-8"))
            if isinstance(spec, dict) and isinstance(spec.get("dataset"), dict):
                dataset = spec["dataset"]
        payload_json.write_text(json.dumps({"dataset": dataset}, indent=2), encoding="utf-8")
