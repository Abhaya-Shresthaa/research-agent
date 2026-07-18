from __future__ import annotations

import datetime
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dynamic_cloud.dataset_config import normalize_dataset_config


def _timestamp() -> str:
    """Stable YYYYMMDD-HHMMSS stamp for archive sibling names."""
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def _run_stamp() -> str:
    """Compact ``MM-DD-HHMM`` stamp used in per-execution run folder names."""
    return datetime.datetime.now().strftime("%m-%d-%H%M")


def archive_existing_dir(path: Path) -> Path | None:
    """Rename an existing directory to a timestamped archive sibling so its
    contents survive a re-run with the same job_id. Returns the archive path,
    or ``None`` if ``path`` did not exist. **Never destroys data** — the only
    operation is a same-filesystem rename into ``<name>.<ts>.archive``.

    Used instead of ``shutil.rmtree`` so that re-running a job with the same
    title preserves the prior completed run's outputs / prepared workspace
    (Section E, E1) instead of silently deleting them.
    """
    if not path.exists():
        return None
    base = f"{path.name}.{_timestamp()}.archive"
    archive = path.parent / base
    # If two archives land in the same second, disambiguate with a counter.
    i = 0
    while archive.exists():
        i += 1
        archive = path.parent / f"{base}{i}"
    # Same-filesystem rename (path and its archive are siblings).
    shutil.move(str(path), str(archive))
    return archive


def _short_slug(label: str) -> str:
    """Filesystem-safe slug for a run folder name.

    The cloud workflow passes the LLM-generated ``job_title`` (already short);
    research-only runs pass the raw query, which is collapsed to the first two
    tokens (and then length-capped) when it is too long so the folder name
    stays compact. Blank or symbol-only input falls back to ``run``.
    """
    slug = safe_job_id((label or "").strip()).strip(".-")
    # safe_job_id falls back to "dynamic-job" for empty/symbol-only input.
    if not slug or slug == "dynamic-job":
        return "run"
    if len(slug) > 30:
        slug = "-".join(slug.split("-")[:2]).strip(".-")
    if len(slug) > 30:
        slug = slug[:30].strip(".-")
    return slug or "run"


@dataclass(frozen=True)
class JobWorkspace:
    job_id: str
    root: Path
    payload_dir: Path
    generated_dir: Path
    data_dir: Path
    outputs_dir: Path


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


def create_run_dir(outputs_root: Path, label: str) -> Path:
    """Create and return a per-execution folder under ``outputs_root``.

    The folder name is ``<slug>-<MM-DD-HHMM>`` where ``slug`` is a short,
    filesystem-safe rendering of ``label`` — the LLM-generated ``job_title``
    for cloud runs (see ``_short_slug``), or a condensed form of the query for
    research-only runs. The minute-precision stamp makes the name unique per
    minute; if a same-minute collision happens, a ``-2``, ``-3`` … suffix is
    appended.

    Everything one execution produces — the research ``report.md``, the cloud
    experiment ``outputs/``, the generated payload under ``runtime_generated/``,
    and the unified ``final-report.md`` — lands inside this single folder.
    """
    slug = _short_slug((label or "").strip())
    ts = _run_stamp()
    name = f"{slug}-{ts}"
    run_dir = outputs_root / name
    suffix = 2
    while run_dir.exists():
        run_dir = outputs_root / f"{name}-{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def prepare_workspace(spec: dict[str, Any], *, run_dir: Path, reset: bool = False) -> JobWorkspace:
    """Prepare the generated payload workspace directly under ``run_dir``.

    The workspace root is ``run_dir / "runtime_generated"`` (holding ``payload/``
    and ``job_spec.json``), and the cloud experiment outputs directory is
    ``run_dir / "outputs"`` — so generation and downloads land in the final
    per-execution folder with no post-run shifting.
    """
    job_id = safe_job_id(str(spec["job_id"]))
    spec["dataset"] = normalize_dataset_config(spec.get("dataset") or {})
    root = run_dir / "runtime_generated"
    payload_dir = root / "payload"
    generated_dir = payload_dir / "generated"
    data_dir = payload_dir / "data"
    outputs_dir = run_dir / "outputs"

    if reset and root.exists():
        # E1: a re-run with the same job_id must not silently wipe a prior
        # prepared workspace. Archive it to a timestamped sibling instead of
        # rmtree'ing it, so the prior generated payload/spec survives.
        archived = archive_existing_dir(root)
        if archived:
            print(f"  Prior workspace archived to {archived.name} (preserved).")

    generated_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    (root / "job_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")

    prepare_dataset(spec.get("dataset") or {}, data_dir)
    return JobWorkspace(
        job_id=job_id,
        root=root,
        payload_dir=payload_dir,
        generated_dir=generated_dir,
        data_dir=data_dir,
        outputs_dir=outputs_dir,
    )


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
