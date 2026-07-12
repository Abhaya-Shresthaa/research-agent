from __future__ import annotations

import argparse
import os
from pathlib import Path

from dynamic_cloud.config import load_amd_settings, load_llm_settings, load_environment

load_environment()

from dynamic_cloud.runtime_layers import write_layered_payload
from dynamic_cloud.vm_options import validate_vm_accelerator, vm_option_for_size
from dynamic_cloud.workspace import (
    load_job_spec,
    prepare_workspace,
    validate_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a dynamic Docker ML job on an AMD GPU Droplet.")
    parser.add_argument("--spec", required=True, type=Path, help="Path to the runtime job spec JSON.")
    parser.add_argument("--generate", action="store_true", help="Use an LLM to generate structured runtime metadata and script.py.")
    parser.add_argument("--execute", action="store_true", help="Actually create AMD GPU Droplet and run the job.")
    parser.add_argument("--validate", action="store_true", help="Validate an already generated or hand-written payload.")
    parser.add_argument("--reset", action="store_true", help="Recreate the local runtime workspace for this job.")
    parser.add_argument("--keep-vm", action="store_true", help="Keep AMD Droplet after the run for debugging (billing continues).")
    parser.add_argument("--vm-name", help="Override the AMD Droplet name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = load_job_spec(args.spec.resolve())
    workspace = prepare_workspace(spec, reset=args.reset)

    if args.generate:
        from dynamic_cloud.llm_generator import generate_payload

        print("Generating layered runtime payload with LLM...")
        payload = generate_payload(spec, load_llm_settings())
        payload.setdefault("job_spec", spec)
        payload["job_spec"].setdefault("job_id", spec.get("job_id"))
        payload["job_spec"].setdefault("dataset", spec.get("dataset", {}))
        write_layered_payload(workspace, payload)
        spec = payload["job_spec"]

    if args.generate or args.execute or args.validate:
        validate_payload(workspace)
    print(f"Runtime payload ready: {workspace.payload_dir}")
    job_spec_path = workspace.root / "job_spec.json"
    if job_spec_path.exists():
        spec = load_job_spec(job_spec_path)

    if not args.execute:
        print("Local preparation complete. Re-run with --generate, --validate, or --execute for the next stage.")
        return

    from dynamic_cloud.executor import execute_workspace_on_amd

    amd_settings = _load_runner_amd_settings(spec, args.vm_name)
    selected_vm = vm_option_for_size(str(amd_settings.vm_sizes[0]), spec)
    validate_vm_accelerator(
        selected_vm,
        str((spec.get("runtime") or {}).get("accelerator") or spec.get("accelerator") or "gpu"),
    )
    execute_workspace_on_amd(workspace, spec, amd_settings, keep_vm=args.keep_vm)

    print("Done.")


def _load_runner_amd_settings(spec: dict, vm_name_override: str | None):
    from dataclasses import replace

    vm_size = str(spec.get("amd_vm_size") or (spec.get("runtime") or {}).get("amd_vm_size") or "")
    vm_name = vm_name_override or spec.get("job_id", "dynamic-ml-droplet")[:55]

    amd_settings = load_amd_settings(vm_name=vm_name)
    if not vm_size:
        return amd_settings
    return replace(amd_settings, vm_sizes=(vm_size,))


if __name__ == "__main__":
    main()
