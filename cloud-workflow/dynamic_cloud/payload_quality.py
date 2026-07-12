from __future__ import annotations

import ast
import json
import re
from typing import Any

from dynamic_cloud.runtime_layers import (
    _load_outputs_helper_functions,
    imports_from_python,
    normalize_accelerator,
    normalize_framework,
)


REVIEW_PROMPT = """
You are the second LLM in a guarded AMD GPU Cloud ML job generation pipeline.

Review the generated payload against the original user requirements, selected runtime, and static analyzer report.
Return strict JSON only:
{
  "approved": true,
  "issues": [
    {"severity": "error or warning", "message": "..."}
  ],
  "required_changes": [],
  "notes": []
}

Approval rules:
- Approve only if generated/script.py satisfies the user requirements and selected framework/accelerator.
- Reject if requested details such as framework, epochs, dataset handling, outputs, or metrics are missing or contradicted.
- When selected_dataset_metadata is present in the context, reject if the script assumes different split folders, class names, label files, or Hugging Face feature names than the inspected metadata provides.
- Reject if Dockerfile or requirements.txt are generated for the normal layered workflow.
- Reject if generated/script.py contains dataset download logic. Dataset downloads are handled only by /workspace/dataset_manager.py from payload.json metadata.
- Reject if the script is a training job (detected via .fit(), .backward(), optimizer.step(), train_loader, or epoch parameters) but does NOT call save_model() from the outputs helper. Without save_model(), model.pth is never written and evaluation crashes with FileNotFoundError.
- Reject if the script has syntax errors, placeholder code, missing /outputs writes, or obvious runtime mistakes.
- Reject if script imports from the outputs helper any function NOT in this exact list: save_metrics, log_epoch, save_training_history, save_plot, save_model, save_environment.
- Reject TensorFlow image pipelines that use Dataset.from_generator with output_types instead of output_signature; tf.image.resize needs known image rank/shape.
- Reject attempts to import ImageFolder from torch.utils.data — it does NOT exist there. The correct import is: from torchvision.datasets import ImageFolder.
- Prefer precise, actionable issues. Do not rewrite code in this response.
"""


FIX_PROMPT = """
You repair generated AMD GPU Cloud ML job payloads after static analysis and LLM review.
Return the full corrected payload as strict JSON only, with this shape:
{
  "job_spec": {
    "job_id": "safe-kebab-case-id",
    "objective": "...",
    "task_type": "...",
    "framework": "pytorch or tensorflow",
    "accelerator": "cpu or gpu",
    "extra_packages": [],
    "dataset": {
    "type": "none | local | huggingface | github",
      "id": "source identifier when applicable",
      "repo": "Hugging Face repo when applicable",
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
  "files": {
    "generated/script.py": "..."
  },
  "config": {},
  "summary": {
    "task": "...",
    "dataset_plan": "...",
    "training_plan": "...",
    "estimated_runtime_minutes": 45,
    "outputs": [],
    "notes": []
  }
}

Estimate the runtime realistically:
- Compute estimated_runtime_minutes based on epochs, dataset size, model complexity, and the AMD GPU hardware from the context.
- Default to a reasonable first-run estimate. Do not set 45 as a default — calculate it from the actual job parameters.

Hyperparameter repair contract:
- Put user-selected training controls near the top of generated/script.py as simple constants with literal values:
  EPOCHS = <requested integer>
  BATCH_SIZE = <requested integer>
  LEARNING_RATE = <requested float>
- Use those constants in the actual training call or loop. For TensorFlow/Keras, call model.fit(..., epochs=EPOCHS, batch_size=BATCH_SIZE). For PyTorch, loop over range(EPOCHS).
- Include the same values in save_environment(extra={"hyperparameters": ...}).
- If static analysis says the requested epoch count is missing, add or correct a visible EPOCHS/NUM_EPOCHS constant with the exact requested integer and use it in training.

Only these exact functions exist in the /workspace/generated/outputs.py helper — do NOT invent any others:
  save_metrics(metrics: dict) -> Path
  log_epoch(epoch: int, metrics: dict, log_file="training_logs.json") -> Path
  save_training_history(history, file_name="training_logs.json") -> Path
  save_plot(figure, name: str) -> Path
  save_model(model, path="model", framework="pytorch") -> Path  — saves model file, returns the OUTPUTS directory so os.path.join(result, "model.pth") works
  save_environment(extra=None) -> Path

Hard rules:
- Generate only generated/script.py for the normal layered workflow.
- Do not generate Dockerfile or requirements.txt unless the request explicitly requires a custom OS/container layer.
- Use the selected framework and accelerator exactly.
- Preserve explicit user requirements such as epoch count, model family, dataset, metrics, and output artifacts.
- Put only dataset source metadata in job_spec.dataset. Use type/id/repo/url/local_paths; do not put download commands in generated/script.py.
- Use DATASET_PATH from the environment for prepared datasets. Local data is copied to /workspace/data; remote data is prepared under /workspace/prepared_datasets.
- If selected_dataset_metadata exists in the context, treat it as ground truth for the real dataset tree, split names, class directories, file extensions, label files, and Hugging Face features. Preserve actual names.
- For Hugging Face datasets, generated/script.py may call datasets.load_from_disk(os.environ["DATASET_PATH"]), but must not call datasets.load_dataset.
- For TensorFlow image datasets built with tf.data.Dataset.from_generator, use output_signature with an image TensorSpec rank of 3, for example
  (tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8), tf.TensorSpec(shape=(), dtype=tf.int32)).
  Convert PIL images to RGB in the generator, yield np.asarray(...), then resize/cast in map. Do not use output_types for image generators.
- Use /outputs for all artifacts.
- Import and use /workspace/generated/outputs.py helpers — only the six functions listed above.
- Output only valid JSON. No Markdown.
"""


def build_review_request(
    requirement_context: dict[str, Any],
    payload: dict[str, Any],
    static_report: dict[str, Any],
) -> str:
    return json.dumps(
        {
            "requirement_context": requirement_context,
            "generated_payload": payload,
            "static_analyzer_report": static_report,
        },
        indent=2,
    )


def build_fix_request(
    requirement_context: dict[str, Any],
    payload: dict[str, Any],
    static_report: dict[str, Any],
    review_report: dict[str, Any],
) -> str:
    return json.dumps(
        {
            "requirement_context": requirement_context,
            "previous_payload": payload,
            "static_analyzer_report": static_report,
            "llm_review_report": review_report,
        },
        indent=2,
    )


def _check_outputs_imports(
    script: str,
    issues: list[dict[str, str]],
    warnings: list[dict[str, str]],
) -> None:
    valid = _load_outputs_helper_functions()
    try:
        tree = ast.parse(script)
    except SyntaxError:
        return
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ImportFrom) and node.module in ("outputs", "generated.outputs"):
            for alias in node.names:
                if alias.name not in valid:
                    issues.append(
                        _issue("error", f"Script imports '{alias.name}' from outputs helper, which does not exist. Available: {', '.join(sorted(valid))}.")
                    )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("outputs.") or alias.name.startswith("generated.outputs."):
                    used = alias.asname or alias.name.rsplit(".", 1)[-1]
                    if used not in valid:
                        issues.append(
                            _issue("error", f"Script references '{alias.name}' which does not exist in outputs helper. Available: {', '.join(sorted(valid))}.")
                        )


def _is_training_script(script: str) -> bool:
    """Heuristic: does the generated script contain training code patterns?"""
    lowered = script.lower()
    if any(p in lowered for p in (
        ".fit(",            # TensorFlow/Keras model.fit()
        ".backward(",       # PyTorch loss.backward()
        "optimizer.step(",  # PyTorch optimizer step
        "model.train()",    # PyTorch train mode
        "train_loader",     # PyTorch training DataLoader
        "train_dataset",    # training dataset reference
        "num_epochs",       # epoch-related variable
    )):
        return True
    return False


COMMON_IMPORT_BLACKLIST = {
    "torch.utils.data.ImageFolder": (
        "torch.utils.data does NOT export ImageFolder. Import from torchvision.datasets instead: "
        "from torchvision.datasets import ImageFolder"
    ),
}


def _check_common_import_errors(
    script: str,
    issues: list[dict[str, str]],
) -> None:
    """Catch known LLM-hallucinated imports that cause hard runtime errors on AMD."""
    try:
        tree = ast.parse(script)
    except SyntaxError:
        return
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module is None:
            continue
        for alias in node.names:
            full_name = f"{node.module}.{alias.name}"
            msg = COMMON_IMPORT_BLACKLIST.get(full_name)
            if msg:
                issues.append(_issue("error", msg))


def _validate_save_model_call(
    script: str,
    job_spec: dict[str, Any],
    requirement_context: dict[str, Any],
    issues: list[dict[str, str]],
    warnings: list[dict[str, str]],
) -> None:
    """Check that training jobs call save_model() — without it model.pth is missing at eval time."""
    task_type = str(job_spec.get("task_type") or "").lower()
    requested_epochs = _requested_epochs(requirement_context)
    expected_framework = _selected_framework(requirement_context, job_spec)

    is_training = (
        requested_epochs is not None
        or any(token in task_type for token in ("train", "finetune", "fine-tune"))
        or _is_training_script(script)
    )
    if not is_training:
        return

    if "save_model(" not in script:
        ctx = ""
        if requested_epochs is not None:
            ctx += f", {requested_epochs} epochs"
        if task_type:
            ctx += f" task_type='{task_type}'"
        issues.append(_issue("error",
            f"Training job detected ({ctx.lstrip(', ')}) but generated/script.py does not call "
            f"save_model() from the outputs helper. The generated code MUST invoke "
            f"save_model(model) after training to persist the trained model; without it "
            f"model.pth will be missing and the evaluation section will crash with "
            f"FileNotFoundError."))


def analyze_generated_payload(
    payload: dict[str, Any],
    requirement_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    requirement_context = requirement_context or {}
    issues: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []

    job_spec = payload.get("job_spec")
    files = payload.get("files")
    if not isinstance(job_spec, dict):
        issues.append(_issue("error", "Generated payload is missing a job_spec object."))
        job_spec = {}
    if not isinstance(files, dict):
        issues.append(_issue("error", "Generated payload is missing a files object."))
        files = {}

    generated_names = set(files)
    custom_container = _truthy((job_spec.get("runtime") or {}).get("custom_container")) or _truthy(job_spec.get("custom_container"))
    forbidden = {"Dockerfile", "requirements.txt"} & generated_names
    if forbidden and not custom_container:
        issues.append(
            _issue(
                "error",
                "Layered workflow should not generate per-job Dockerfile or requirements.txt: "
                + ", ".join(sorted(forbidden)),
            )
        )

    script = str(files.get("generated/script.py") or files.get("script.py") or "")
    if not script.strip():
        issues.append(_issue("error", "Generated payload must include non-empty generated/script.py."))
    else:
        lowered = script.lower()
        if "placeholder" in lowered or "no runtime script was generated" in lowered:
            issues.append(_issue("error", "generated/script.py still contains placeholder text."))
        try:
            ast.parse(script)
        except SyntaxError as exc:
            issues.append(_issue("error", f"generated/script.py has a syntax error: {exc}."))

        imports = imports_from_python(script)
        expected_framework = _selected_framework(requirement_context, job_spec)
        if expected_framework == "tensorflow" and not ({"tensorflow", "keras"} & imports):
            issues.append(_issue("error", "Selected framework is TensorFlow, but script does not import tensorflow or keras."))
        if expected_framework == "pytorch" and "torch" not in imports:
            issues.append(_issue("error", "Selected framework is PyTorch, but script does not import torch."))

        requested_epochs = _requested_epochs(requirement_context)
        if requested_epochs is not None and not _script_mentions_epoch_count(script, requested_epochs):
            issues.append(_issue("error", f"User requested {requested_epochs} epochs, but the script does not clearly use that epoch count."))

        _check_outputs_imports(script, issues, warnings)
        _check_dataset_contract(script, job_spec, issues, warnings)
        _check_tf_data_antipatterns(script, issues, warnings)
        _check_common_import_errors(script, issues)
        _validate_save_model_call(script, job_spec, requirement_context, issues, warnings)

        if "/outputs" not in script and "save_metrics" not in script and "save_environment" not in script:
            issues.append(_issue("error", "Script does not appear to write artifacts to /outputs or use the outputs helper."))
        if "metrics.json" not in script and "save_metrics" not in script:
            warnings.append(_issue("warning", "Script does not clearly save metrics.json."))
        if "environment.json" not in script and "save_environment" not in script:
            warnings.append(_issue("warning", "Script does not clearly save environment.json."))

    runtime = job_spec.get("runtime") if isinstance(job_spec.get("runtime"), dict) else {}
    selected_framework = _selected_framework(requirement_context, job_spec)
    selected_accelerator = _selected_accelerator(requirement_context, job_spec)
    if selected_framework and normalize_framework(str(runtime.get("framework") or job_spec.get("framework") or "")) != selected_framework:
        issues.append(_issue("error", f"job_spec/runtime framework must be {selected_framework}."))
    if selected_accelerator and normalize_accelerator(str(runtime.get("accelerator") or job_spec.get("accelerator") or "")) != selected_accelerator:
        issues.append(_issue("error", f"job_spec/runtime accelerator must be {selected_accelerator}."))
    if runtime and runtime.get("entrypoint") not in {None, "generated/script.py"}:
        issues.append(_issue("error", "runtime.entrypoint must be generated/script.py."))

    return {
        "approved": not issues,
        "issues": issues,
        "warnings": warnings,
    }


def review_approved(static_report: dict[str, Any], review_report: dict[str, Any]) -> bool:
    if static_report.get("issues"):
        return False
    if review_report.get("approved") is not True:
        return False
    return not any(str(issue.get("severity", "")).lower() == "error" for issue in review_report.get("issues", []))


def merge_review_notes(static_report: dict[str, Any], review_report: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    for item in static_report.get("issues", []) + static_report.get("warnings", []):
        message = item.get("message")
        if message:
            notes.append(f"static {item.get('severity', 'issue')}: {message}")
    for item in review_report.get("issues", []):
        message = item.get("message")
        if message:
            notes.append(f"review {item.get('severity', 'issue')}: {message}")
    return notes


def _selected_framework(requirement_context: dict[str, Any], job_spec: dict[str, Any]) -> str:
    value = (
        requirement_context.get("selected_framework")
        or (requirement_context.get("selected_runtime") or {}).get("framework")
        or job_spec.get("framework")
        or (job_spec.get("runtime") or {}).get("framework")
        or ""
    )
    return normalize_framework(str(value)) if value else ""


def _selected_accelerator(requirement_context: dict[str, Any], job_spec: dict[str, Any]) -> str:
    value = (
        requirement_context.get("selected_accelerator")
        or (requirement_context.get("selected_runtime") or {}).get("accelerator")
        or job_spec.get("accelerator")
        or (job_spec.get("runtime") or {}).get("accelerator")
        or ""
    )
    return normalize_accelerator(str(value)) if value else ""


def _requested_epochs(requirement_context: dict[str, Any]) -> int | None:
    texts: list[str] = []
    for key in ("user_requirement", "objective"):
        if requirement_context.get(key):
            texts.append(str(requirement_context[key]))
    answers = requirement_context.get("answers")
    if isinstance(answers, dict):
        texts.extend(str(value) for value in answers.values())
    spec = requirement_context.get("job_spec")
    if isinstance(spec, dict):
        texts.append(json.dumps(spec))

    for text in texts:
        match = re.search(r"\b(\d{1,4})\s*(?:epoch|epochs)\b", text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    if isinstance(answers, dict):
        for key, value in answers.items():
            if "epoch" in str(key).lower():
                match = re.search(r"\b(\d{1,4})\b", str(value))
                if match:
                    return int(match.group(1))
    return None


def _script_mentions_epoch_count(script: str, epochs: int) -> bool:
    escaped = re.escape(str(epochs))
    assignment = rf"\b\w*epochs?\w*\s*(?::\s*[^=\n]+)?=\s*(?:int\(\s*)?[\"']?{escaped}[\"']?\s*\)?\b"
    if re.search(assignment, script, flags=re.IGNORECASE):
        return True
    if re.search(rf"['\"]epochs?['\"]\s*:\s*{escaped}\b", script, flags=re.IGNORECASE):
        return True
    if re.search(rf"\brange\(\s*(?:1\s*,\s*)?{escaped}\s*(?:\+\s*1)?\s*\)", script):
        return True
    if re.search(rf"\.fit\([^)]*epochs\s*=\s*{escaped}\b", script, flags=re.IGNORECASE | re.DOTALL):
        return True

    try:
        tree = ast.parse(script)
    except SyntaxError:
        return False
    epoch_vars = _epoch_constant_names(tree, epochs)
    if not epoch_vars:
        return False
    return _epoch_constant_used_for_training(tree, epoch_vars)


def _epoch_constant_names(tree: ast.AST, epochs: int) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        target_names: list[str] = []
        value: ast.AST | None = None
        if isinstance(node, ast.Assign):
            target_names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target_names = [node.target.id]
            value = node.value
        if not target_names or value is None:
            continue
        if not any("epoch" in name.lower() for name in target_names):
            continue
        if _literal_int(value) == epochs:
            names.update(target_names)
    return names


def _epoch_constant_used_for_training(tree: ast.AST, epoch_vars: set[str]) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if _call_name(node.func).endswith(".fit") or _call_name(node.func) == "fit":
                for keyword in node.keywords:
                    if keyword.arg == "epochs" and isinstance(keyword.value, ast.Name) and keyword.value.id in epoch_vars:
                        return True
            if isinstance(node.func, ast.Name) and node.func.id == "range":
                for arg in node.args:
                    if isinstance(arg, ast.Name) and arg.id in epoch_vars:
                        return True
                    if isinstance(arg, ast.BinOp) and _name_in_node(arg, epoch_vars):
                        return True
    return False


def _literal_int(node: ast.AST) -> int | None:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, int):
            return node.value
        if isinstance(node.value, str) and node.value.strip().isdigit():
            return int(node.value.strip())
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "int" and len(node.args) == 1:
        return _literal_int(node.args[0])
    return None


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _name_in_node(node: ast.AST, names: set[str]) -> bool:
    return any(isinstance(child, ast.Name) and child.id in names for child in ast.walk(node))


def _check_dataset_contract(
    script: str,
    job_spec: dict[str, Any],
    issues: list[dict[str, str]],
    warnings: list[dict[str, str]],
) -> None:
    dataset = job_spec.get("dataset") if isinstance(job_spec.get("dataset"), dict) else {}
    dataset_type = str(dataset.get("type") or "").strip().lower()
    lowered = script.lower()
    forbidden_patterns = {
        "datasets.load_dataset": "Hugging Face downloads must happen in dataset_manager.py; use load_from_disk(DATASET_PATH) in script.py.",
        "load_dataset(": "Hugging Face downloads must happen in dataset_manager.py; use load_from_disk(DATASET_PATH) in script.py.",
        "git clone": "GitHub clones must happen in dataset_manager.py.",
    }
    for pattern, message in forbidden_patterns.items():
        if pattern in lowered:
            issues.append(_issue("error", f"generated/script.py contains dataset download logic ({pattern.strip()}): {message}"))

    if dataset_type and dataset_type != "none" and "dataset_path" not in lowered and "data_dir" not in lowered and "os.environ" not in lowered:
        warnings.append(_issue("warning", "Script does not clearly read DATASET_PATH or the prepared dataset directory."))


def _check_tf_data_antipatterns(
    script: str,
    issues: list[dict[str, str]],
    warnings: list[dict[str, str]],
) -> None:
    if (
        "tf.data.Dataset.from_generator" in script
        and "output_types" in script
        and "tf.image.resize" in script
    ):
        issues.append(_issue(
            "error",
            "TensorFlow image generators must use output_signature with a rank-3 image TensorSpec before tf.image.resize; "
            "from_generator(..., output_types=...) leaves image shape unknown and can fail with \"images contains no shape\".",
        ))

    if ".map(" not in script or "np.array(" not in script:
        return

    lines = script.split("\n")
    for lineno, line in enumerate(lines, start=1):
        if "np.array(" not in line:
            continue
        for lookback in range(max(0, lineno - 6), lineno):
            prev = lines[lookback]
            if ".map(" in prev or ".map(lambda" in prev:
                issues.append(_issue(
                    "error",
                    f"generated/script.py:{lineno}: np.array() inside tf.data.Dataset.map() "
                    f"fails on symbolic tensors. Convert to numpy in the generator instead.",
                ))
                return


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _issue(severity: str, message: str) -> dict[str, str]:
    return {"severity": severity, "message": message}
