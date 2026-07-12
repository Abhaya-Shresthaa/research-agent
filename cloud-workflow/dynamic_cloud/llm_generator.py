from __future__ import annotations

from typing import Any

from dynamic_cloud.config import LlmSettings
from dynamic_cloud.llm_client import generate_json_with_model
from dynamic_cloud.payload_quality import (
    FIX_PROMPT,
    REVIEW_PROMPT,
    analyze_generated_payload,
    build_fix_request,
    build_review_request,
    merge_review_notes,
    review_approved,
)


SYSTEM_PROMPT = """You generate portable ML job payloads for a layered AMD GPU Cloud Docker workflow.
Return only strict JSON with this shape:
{
  "job_spec": {
    "job_id": "...",
    "objective": "...",
    "framework": "pytorch or tensorflow",
    "accelerator": "cpu or gpu",
    "extra_packages": [],
    "runtime": {
      "entrypoint": "generated/script.py",
      "outputs_dir": "/outputs",
      "workspace_dir": "/workspace"
    }
  },
  "files": {
    "generated/script.py": "..."
  }
}

Standardized output schema — ALL listed artifacts are **REQUIRED**; the generated script MUST call the appropriate save function for each one, or the job will fail validation:
  environment.json      — required for every job; call save_environment()
  metrics.json          — required for training/eval jobs; call save_metrics()
  training_logs.json    — required for training jobs; call log_epoch() or save_training_history()
  model.pth / model.keras / model.joblib — **REQUIRED** for training jobs; call save_model() — the validation gate rejects if missing
  logs/                 — required; write runtime logs here (save_plot() auto-saves to plots/)
  plots/                — required for training jobs; use save_plot()

Use the pre-bundled /workspace/generated/outputs.py helper. Only these exact functions exist — do NOT invent any others:
  save_metrics(metrics: dict) -> Path
  log_epoch(epoch: int, metrics: dict, log_file="training_logs.json") -> Path
  save_training_history(history, file_name="training_logs.json") -> Path
  save_plot(figure, name: str) -> Path  — pass plt.gcf() or any matplotlib figure; saves to plots/ automatically
  save_model(model, path="model", framework="pytorch") -> Path  — saves model file, returns the OUTPUTS directory so os.path.join(result, "model.pth") works
  save_environment(extra=None) -> Path

Your generated script MUST import only from the list above. Any hallucinated import will cause a runtime error and waste AMD GPU Cloud resources.

CRITICAL — paths in generated code:
- NEVER hardcode absolute paths like '/outputs/' in generated/script.py. The actual output directory is resolved at runtime via the DYNAMIC_CLOUD_OUTPUTS_DIR environment variable and may be different.
- Instead, always either:
  (a) Use the outputs.py helper functions listed above (save_metrics, log_epoch, etc.) — they resolve the correct path automatically.
  (b) Import OUTPUTS from outputs:  from outputs import OUTPUTS  then use  OUTPUTS / "logs/training.log"  (not '/outputs/logs/training.log').
  (c) Read the env var:  outputs_dir = os.environ.get('DYNAMIC_CLOUD_OUTPUTS_DIR', 'outputs')  then  os.path.join(outputs_dir, 'logs/training.log').
- A hardcoded absolute path like '/outputs/logs/training.log' will crash because that directory does not exist on the remote file system.
- Do NOT use deprecated PyTorch arguments removed in v2.4+. In particular, do NOT pass verbose=True to ReduceLROnPlateau, StepLR, or any LR scheduler — the verbose parameter was removed in PyTorch 2.4 and will crash with TypeError.
- For image folder datasets, import ImageFolder from torchvision.datasets (NOT from torch.utils.data). torch.utils.data does NOT contain ImageFolder in any modern PyTorch — it lives in torchvision.datasets.ImageFolder.

Rules:
- Do not generate Dockerfile or requirements.txt; extra packages belong in job_spec.extra_packages/runtime.extra_packages.
- The orchestrator copies the generated payload into the pulled Docker container at /workspace, with optional data at /workspace/data.
- The container must write all artifacts to the OUTPUTS directory (resolved via DYNAMIC_CLOUD_OUTPUTS_DIR env var at runtime) using the standard schema above. NEVER hardcode '/outputs/...' absolute paths in generated code.
- Put artifacts that must exist in job_spec.expected_outputs. Do not mark plots or training logs as expected for non-training jobs unless the user explicitly needs them.
- The script must be generic to the job spec and must not assume dog_vs_cat names unless explicitly requested.
- If data is optional or missing, create a tiny synthetic/demo path or raise a clear actionable error.
- Keep generated code self-contained and executable.
"""


def generate_payload(spec: dict[str, Any], settings: LlmSettings) -> dict[str, Any]:
    from model import make_model

    model_handle = make_model(settings)
    payload = _generate_initial_payload(model_handle, spec)
    return _review_and_fix_payload(model_handle, payload, spec)


def _generate_initial_payload(model_handle: Any, spec: dict[str, Any]) -> dict[str, Any]:
    payload = generate_json_with_model(
        model_handle,
        SYSTEM_PROMPT,
        {"instruction": "Create the runtime Docker payload for this job spec.", "job_spec": spec},
    )
    if not isinstance(payload.get("files"), dict):
        raise ValueError("LLM response must contain a files object.")

    return payload


def _review_and_fix_payload(
    model_handle: Any,
    payload: dict[str, Any],
    spec: dict[str, Any],
    max_fix_attempts: int = 2,
) -> dict[str, Any]:
    context = {
        "job_spec": spec,
        "objective": spec.get("objective"),
        "selected_framework": spec.get("framework") or (spec.get("runtime") or {}).get("framework"),
        "selected_accelerator": spec.get("accelerator") or (spec.get("runtime") or {}).get("accelerator"),
    }

    for attempt in range(max_fix_attempts + 1):
        static_report = analyze_generated_payload(payload, context)
        review_report = _review_payload(model_handle, context, payload, static_report)
        if review_approved(static_report, review_report):
            payload.setdefault("quality_gate", {})
            payload["quality_gate"] = {
                "static_analyzer": static_report,
                "llm_review": review_report,
                "fix_attempts": attempt,
            }
            return payload

        if attempt >= max_fix_attempts:
            notes = "\n".join(merge_review_notes(static_report, review_report))
            raise ValueError(f"Generated payload failed quality review after {max_fix_attempts} fix attempts:\n{notes}")

        payload = _fix_payload(model_handle, context, payload, static_report, review_report)

    raise RuntimeError("Unreachable payload quality loop exit.")


def _review_payload(
    model_handle: Any,
    context: dict[str, Any],
    payload: dict[str, Any],
    static_report: dict[str, Any],
) -> dict[str, Any]:
    review = generate_json_with_model(
        model_handle,
        REVIEW_PROMPT,
        build_review_request(context, payload, static_report),
    )
    if not isinstance(review.get("issues", []), list):
        review["issues"] = []
    return review


def _fix_payload(
    model_handle: Any,
    context: dict[str, Any],
    payload: dict[str, Any],
    static_report: dict[str, Any],
    review_report: dict[str, Any],
) -> dict[str, Any]:
    fixed = generate_json_with_model(
        model_handle,
        FIX_PROMPT,
        build_fix_request(context, payload, static_report, review_report),
    )
    if not isinstance(fixed.get("files"), dict):
        raise ValueError("LLM repair response must contain a files object.")
    return fixed
