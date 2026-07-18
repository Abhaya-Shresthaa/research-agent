#!/usr/bin/env python3
"""
Unified Orchestrator — Deep Research Agent + Cloud Experiment Agent

Brings two standalone workflows into one CLI entry point without modifying
any original project files, functions, or directory structures.

Usage:
    python main.py

Modes:
    1. Research Agent       – Deep web research, question feedback, Markdown report
    2. Cloud Experiment Agent – ML job config, AMD VM infrastructure, remote execution
    3. Both Workflows       – Sequential questioning → parallel execution
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Path setup — add both workflow directories to sys.path BEFORE imports
# ---------------------------------------------------------------------------

_BASE_DIR = Path(__file__).resolve().parent
_RESEARCH_DIR = _BASE_DIR / "research-workflow" / "src"
_CLOUD_DIR = _BASE_DIR / "cloud-workflow"

# Insert so that "from src.deep_research import ..." and "import run" work
if str(_RESEARCH_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_RESEARCH_DIR.parent))
if str(_CLOUD_DIR) not in sys.path:
    sys.path.insert(0, str(_CLOUD_DIR))

# ---------------------------------------------------------------------------
# 2. Environment loading — load centralized .env from final_amd/
# ---------------------------------------------------------------------------

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(path=None, override=False):
        return False

_env_path = _BASE_DIR / ".env"
if _env_path.exists():
    load_dotenv(str(_env_path), override=True)

# ---------------------------------------------------------------------------
# 3. Imports — research API + cloud module
# ---------------------------------------------------------------------------

# Research workflow imports
from src.deep_research import deep_research, write_final_answer, write_final_report
from src.feedback import generate_feedback
from src.ai.providers import get_model_id as _research_model_id

# Cloud workflow — import as a module to access private functions
import run as _cloud_run

# Re-export all cloud private helpers as module-level aliases
_cloud_framework = _cloud_run._select_framework
_cloud_accelerator = _cloud_run._select_accelerator
_cloud_dataset_source = _cloud_run._select_dataset_source
_cloud_llm_questions = _cloud_run._ask_llm_for_questions
_cloud_ask_questions = _cloud_run._ask_questions
_cloud_select_vm = _cloud_run._select_vm
_cloud_generate_payload = _cloud_run._generate_runtime_payload
_cloud_inspect_dataset_amd = _cloud_run._inspect_dataset_on_amd
_cloud_confirm = _cloud_run._confirm
_cloud_execute_on_amd = _cloud_run._execute_on_amd
_cloud_cleanup_vm = _cloud_run._cleanup_selected_vm
_cloud_print_summary = _cloud_run._print_summary
_cloud_estimate_cost = _cloud_run._estimate_cost
_cloud_format_duration = _cloud_run._format_duration
_cloud_load_amd_settings = _cloud_run._load_selected_amd_settings
_cloud_setup_local_dataset = _cloud_run._setup_and_inspect_local_dataset
_cloud_apply_existing_droplet = _cloud_run._apply_existing_droplet_selection
_cloud_check_user_script = _cloud_run._check_user_script

# Cloud public helpers
from dynamic_cloud.runtime_layers import select_image, normalize_framework, validate_local_image_support
from dynamic_cloud.workspace import prepare_workspace, safe_job_id, validate_payload, normalize_dataset_config, create_run_dir
from dynamic_cloud.runtime_layers import write_layered_payload
from dynamic_cloud.config import AmdDropletSession
from dynamic_cloud.vm_options import select_amd_image, validate_vm_accelerator, vm_option_for_size
from dynamic_cloud.executor import _generate_final_report
import json
from typing import Any

# ---------------------------------------------------------------------------
# 4. Monkey-patch research verbose logging (cosmetic)
# ---------------------------------------------------------------------------

import src.deep_research as _deep_module

def _deep_quiet_log(*args, **kwargs):
    """Suppress all internal deep-research per-query logs."""
    pass


_deep_module._log = _deep_quiet_log

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _ask(question: str) -> str:
    try:
        return input(question)
    except EOFError:
        return ""


def _log(*args, **kwargs):
    pass


# ---------------------------------------------------------------------------
# 5. Research agent — standalone interactive (Option 1)
# ---------------------------------------------------------------------------


def _log_research_start(model_id: str) -> None:
    _log(f"\n{'=' * 56}")
    _log(f"  Deep Research Agent")
    _log(f"  Model: {model_id}")
    _log(f"{'=' * 56}\n")


async def _run_research_workflow(
    initial_query: str,
    breadth: int,
    depth: int,
    is_report: bool,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    """Execute the deep research pipeline and return results.

    When ``run_dir`` is provided, the report/answer is written into that
    per-execution folder; otherwise it falls back to ``_BASE_DIR/outputs/``.
    """
    combined_query = initial_query

    if is_report:
        _log("Creating research plan...")
        follow_up_questions = await generate_feedback(query=initial_query)
        _log("\nTo better understand your research needs, please answer these follow-up questions:")

        answers: list[str] = []
        for question in follow_up_questions:
            answer = _ask(f"\n{question}\nYour answer: ")
            answers.append(answer)

        qa_pairs = "\n".join(
            f"Q: {q}\nA: {a}" for q, a in zip(follow_up_questions, answers)
        )
        combined_query = (
            f"Initial Query: {initial_query}\n"
            f"Follow-up Questions and Answers:\n{qa_pairs}"
        )

    _log("\nStarting research...\n")

    result = await deep_research(
        query=combined_query,
        breadth=breadth,
        depth=depth,
    )

    _log("\nResearch complete.")
    _log("Writing final output...")

    outputs_dir = run_dir if run_dir is not None else (_BASE_DIR / "outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    if is_report:
        report = write_final_report(
            prompt=combined_query,
            learnings=result.learnings,
            visited_urls=result.visited_urls,
            relevant_images=result.relevant_images,
        )
        report_path = outputs_dir / "report.md"
        report_path.write_text(report, encoding="utf-8")
        _log(f"\nReport has been saved to {report_path}")
        return {"type": "report", "path": str(report_path), "learnings": len(result.learnings), "urls": len(result.visited_urls), "run_dir": str(run_dir) if run_dir else None}
    else:
        answer = write_final_answer(
            prompt=combined_query,
            learnings=result.learnings,
        )
        answer_path = outputs_dir / "answer.md"
        answer_path.write_text(answer, encoding="utf-8")
        _log(f"\nAnswer has been saved to {answer_path}")
        return {"type": "answer", "path": str(answer_path), "learnings": len(result.learnings), "urls": len(result.visited_urls), "run_dir": str(run_dir) if run_dir else None}


async def _research_interactive() -> dict[str, Any]:
    """Full interactive Research Agent flow (Option 1)."""
    _log_research_start(_research_model_id())

    initial_query = _ask("What would you like to research? ")
    if not initial_query.strip():
        print("No topic entered. Exiting.")
        return {"status": "cancelled"}

    # Per-execution folder for this run's research output.
    run_dir = create_run_dir(_BASE_DIR / "outputs", initial_query)
    print(f"\nRun folder: {run_dir.relative_to(_BASE_DIR)}")

    breadth_raw = _ask("Enter research breadth (recommended 2-10, default 4): ")
    breadth = int(breadth_raw) if breadth_raw.strip().isdigit() else 4

    depth_raw = _ask("Enter research depth (recommended 1-5, default 2): ")
    depth = int(depth_raw) if depth_raw.strip().isdigit() else 2

    is_report = (
        _ask("Do you want to generate a long report or a specific answer? (report/answer, default report): ")
        != "answer"
    )

    return await _run_research_workflow(initial_query, breadth, depth, is_report, run_dir=run_dir)


# ---------------------------------------------------------------------------
# 6. Cloud agent — standalone interactive (Option 2)
# ---------------------------------------------------------------------------


def _cloud_interactive() -> dict[str, Any]:
    """Full interactive Cloud Experiment Agent flow (Option 2)."""
    print(f"\n{'=' * 56}")
    print(f"  Cloud Experiment Agent")
    print(f"  AMD GPU ML Job Builder")
    print(f"{'=' * 56}\n")

    user_requirement = input("What do you want to train, test, or run?\n> ").strip()
    if not user_requirement:
        print("No requirement entered. Exiting.")
        return {"status": "cancelled"}

    has_user_script, user_script_content = _cloud_check_user_script()
    selected_framework = _cloud_framework()
    selected_accelerator = _cloud_accelerator()
    selected_dataset = _cloud_dataset_source()

    print("Making personalized questions...")
    questions_payload = _cloud_llm_questions(
        user_requirement, selected_framework, selected_accelerator, selected_dataset,
        has_user_script=has_user_script, user_script_content=user_script_content,
    )
    answers = _cloud_ask_questions(questions_payload.get("questions", []))
    selected_vm = _cloud_select_vm()
    image_slug, node_name = select_amd_image(selected_vm["size"])
    selected_vm["gpu_image"] = image_slug
    selected_vm["vm_name"] = node_name
    _cloud_apply_existing_droplet(selected_vm, None, None)
    validate_vm_accelerator(selected_vm, selected_accelerator)

    # Per-execution run folder, named from the LLM job_title (short slug) +
    # <MM-DD-HHMM>. Created after the questions so the title is available.
    run_dir = create_run_dir(
        _BASE_DIR / "outputs",
        str(questions_payload.get("job_title") or user_requirement),
    )
    print(f"\nRun folder: {run_dir.relative_to(_BASE_DIR)}")

    return _cloud_continue_flow(
        user_requirement=user_requirement,
        selected_framework=selected_framework,
        selected_accelerator=selected_accelerator,
        selected_dataset=selected_dataset,
        selected_vm=selected_vm,
        questions_payload=questions_payload,
        answers=answers,
        has_user_script=has_user_script,
        user_script_content=user_script_content,
        run_dir=run_dir,
    )


def _cloud_continue_flow(
    *,
    user_requirement: str,
    selected_framework: str,
    selected_accelerator: str,
    selected_dataset: dict[str, Any],
    selected_vm: dict[str, Any],
    questions_payload: dict[str, Any],
    answers: dict[str, str],
    has_user_script: bool = False,
    user_script_content: str = "",
    run_now: bool | None = None,
    keep_vm_choice: bool | None = None,
    non_interactive: bool = False,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    """Shared post-questions cloud workflow logic (called by option 2 and option 3).

    ``run_now`` / ``keep_vm_choice`` let callers pre-collect the two execution
    decisions (option 3) so the parallel phase never blocks on ``input()`` from
    a worker thread. When ``None`` (option 2), the original interactive
    ``_cloud_confirm`` prompts are used. When ``non_interactive`` is set, the
    dataset-inspection-failure fallback auto-cancels instead of prompting.
    """
    job_id = safe_job_id(str(questions_payload.get("job_title") or "dynamic-job"))
    dataset_metadata: dict[str, Any] | None = None
    inspection_session: AmdDropletSession | None = None

    # ── Helper: destroy an inspection droplet that was created but not yet used ──

    def _destroy_inspection_droplet() -> None:
        """Destroy the inspection Droplet if one was created for this run."""
        nonlocal inspection_session
        if not inspection_session:
            return
        try:
            from dynamic_cloud.amd_droplet import AmdDropletManager
            from dynamic_cloud.config import load_amd_settings

            amd_settings = load_amd_settings(
                vm_name=selected_vm.get("vm_name", "cleanup")
            )
            manager = AmdDropletManager(amd_settings)
            manager.adopt(inspection_session.droplet_id)
            manager.destroy()
            print(f"\nDestroyed inspection Droplet {inspection_session.droplet_id}.")
        except BaseException as exc:
            print(f"\nFailed to auto-cleanup inspection Droplet: {exc}")
            if inspection_session and inspection_session.droplet_id:
                print(
                    f"IMPORTANT: Manually destroy Droplet {inspection_session.droplet_id} "
                    f"to stop billing!"
                )
        finally:
            inspection_session = None

    # --- Dataset inspection ---
    if selected_dataset.get("type") == "local":
        dataset_metadata = _cloud_setup_local_dataset(selected_dataset)
    elif selected_dataset.get("type") != "none":
        try:
            dataset_metadata, selected_vm, inspection_session = _cloud_inspect_dataset_amd(
                job_id,
                user_requirement,
                selected_framework,
                selected_accelerator,
                selected_dataset,
                selected_vm,
                run_dir=run_dir,
            )
        except RuntimeError as exc:
            print(f"\n[WARNING] Dataset inspection failed: {exc}")
            if non_interactive:
                # Parallel/unattended mode (option 3): cannot prompt from a
                # worker thread. Match the prompt's default (False → cancel).
                print("Cloud workflow cancelled (non-interactive: dataset inspection failed).")
                return {"status": "cancelled", "reason": "dataset inspection failed"}
            fallback = _cloud_confirm(
                "Continue with no external dataset instead?", default=False
            )
            if fallback:
                selected_dataset = normalize_dataset_config({"type": "none"})
                dataset_metadata = None
                print("Continuing with no external dataset.")
            else:
                print("Cloud workflow cancelled.")
                return {"status": "cancelled"}

    # --- Payload generation (with droplet cleanup on failure) ---
    try:
        generation_payload = _cloud_generate_payload(
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
    except BaseException:
        _destroy_inspection_droplet()
        raise

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

    workspace = prepare_workspace(job_spec, run_dir=run_dir, reset=dataset_metadata is None)
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

    _cloud_print_summary(job_spec, summary, selected_vm, workspace, metadata)

    # --- Check AMD config ---
    try:
        _cloud_load_amd_settings(workspace, selected_vm)
    except Exception as exc:
        print(f"\nAMD Cloud setup is not ready yet: {exc}")
        print(f"\nPrepared files were still written here: {workspace.payload_dir}")
        print("Fix the AMD configuration, then run this script again.")
        return {"status": "prepared", "workspace": str(workspace.root), "run_dir": str(run_dir) if run_dir else None}

    estimated_minutes = job_spec.get("estimated_runtime_minutes")
    if estimated_minutes:
        estimated_cost = _cloud_estimate_cost(selected_vm, estimated_minutes)
        print(f"\nEstimated runtime: {_cloud_format_duration(estimated_minutes)}")
        if estimated_cost is not None:
            print(f"Estimated AMD GPU cost: ${estimated_cost:.2f}")

    vm_already_running = inspection_session is not None

    run_prompt = (
        "Continue on the same AMD Droplet and run training now?"
        if vm_already_running
        else "Start AMD GPU Droplet and run training now?"
    )
    if run_now is not None:
        proceed = run_now
        print(f"\n{run_prompt} {'yes' if proceed else 'no'} (pre-selected)")
    else:
        proceed = _cloud_confirm(run_prompt)
    if not proceed:
        print(f"\nPrepared files only. You can inspect them here: {workspace.payload_dir}")
        print(f"\nTo execute later run:\n  python run.py --execute-prepared {workspace.root}")
        if inspection_session:
            _cloud_cleanup_vm(workspace, selected_vm, inspection_session)
        return {"status": "prepared", "workspace": str(workspace.root), "run_dir": str(run_dir) if run_dir else None}

    if keep_vm_choice is not None and not vm_already_running:
        keep_vm = keep_vm_choice
        print(f"\nKeep the Droplet after the run? {'yes' if keep_vm else 'no'} (pre-selected)")
    else:
        keep_vm = (
            bool(inspection_session and not inspection_session.created_for_dataset_inspection)
            or (
                False
                if vm_already_running
                else _cloud_confirm(
                    "Keep the Droplet after the run for debugging? (billing continues)",
                    default=False,
                )
            )
        )
    _cloud_execute_on_amd(
        workspace,
        job_spec,
        selected_vm,
        keep_vm,
        preserve_remote=vm_already_running,
        vm_already_selected=vm_already_running,
        session=inspection_session,
    )
    return {
        "status": "completed",
        "job_id": workspace.job_id,
        "workspace": str(workspace.root),
        "run_dir": str(run_dir) if run_dir else None,
    }


# ---------------------------------------------------------------------------
# 7. Both Workflows — sequential questioning → parallel execution (Option 3)
# ---------------------------------------------------------------------------


async def _run_both_workflows() -> dict[str, Any]:
    """Option 3: sequential questioning, then parallel execution."""
    print(f"\n{'=' * 56}")
    print(f"  Unified Orchestrator — Both Workflows")
    print(f"  You will be guided through two phases of questions,")
    print(f"  then both engines will run in parallel.")
    print(f"{'=' * 56}\n")

    # ── Single unified query ──────────────────────────────────────────────
    query = _ask("Enter your primary research topic or training requirement:\n> ").strip()
    if not query:
        print("No input entered. Exiting.")
        return {"status": "cancelled"}

    print(f"\n{'─' * 56}")
    print(f"  Phase 1 — Cloud Agent Questions")
    print(f"{'─' * 56}")

    has_user_script, user_script_content = _cloud_check_user_script()
    selected_framework = _cloud_framework()
    selected_accelerator = _cloud_accelerator()
    selected_dataset = _cloud_dataset_source()

    print("Making personalized questions...")
    questions_payload = _cloud_llm_questions(
        query, selected_framework, selected_accelerator, selected_dataset,
        has_user_script=has_user_script, user_script_content=user_script_content,
    )
    answers = _cloud_ask_questions(questions_payload.get("questions", []))
    selected_vm = _cloud_select_vm()
    image_slug, node_name = select_amd_image(selected_vm["size"])
    selected_vm["gpu_image"] = image_slug
    selected_vm["vm_name"] = node_name
    _cloud_apply_existing_droplet(selected_vm, None, None)
    validate_vm_accelerator(selected_vm, selected_accelerator)

    # Per-execution folder shared by both workflows, named from the LLM
    # job_title (short slug) + <MM-DD-HHMM>. Created after Phase 1 so the
    # title is available. Research writes report.md here, cloud writes
    # runtime_generated/ + outputs/ here, and the unified final-report.md is
    # generated here.
    run_dir = create_run_dir(
        _BASE_DIR / "outputs",
        str(questions_payload.get("job_title") or query),
    )
    print(f"\nRun folder: {run_dir.relative_to(_BASE_DIR)}")

    # ── Execution decisions are made interactively in the cloud flow ──
    # The "Continue on the same AMD Droplet and run training now?" and
    # "Keep the Droplet after the run?" prompts are asked (y/n) at execution
    # time inside _cloud_continue_flow — after dataset inspection, when the
    # cost estimate is known and the droplet exists — rather than pre-collected
    # here. Pre-collecting (with default=no) silently locked the run-now
    # decision to "no" if the user pressed Enter, skipping training. Now the
    # user is asked y/n (default yes); "no" yields the prepared-files-only path.

    print(f"\n{'─' * 56}")
    print(f"  Phase 2 — Research Agent Questions")
    print(f"{'─' * 56}")

    breadth = 4
    depth = 2
    is_report = True

    # Handle research feedback questions
    research_combined_query = query
    if is_report:
        _log("Creating research plan...")
        follow_up_questions = await generate_feedback(query=query)
        _log("\nTo better understand your research needs, please answer these follow-up questions:")

        answers_research: list[str] = []
        for question in follow_up_questions:
            answer = _ask(f"\n{question}\nYour answer: ")
            answers_research.append(answer)

        qa_pairs = "\n".join(
            f"Q: {q}\nA: {a}" for q, a in zip(follow_up_questions, answers_research)
        )
        research_combined_query = (
            f"Initial Query: {query}\n"
            f"Follow-up Questions and Answers:\n{qa_pairs}"
        )

    # ── Phase 3: Parallel execution ───────────────────────────────────────
    print(f"\n{'=' * 56}")
    print(f"  🚀 Both workflows running parallel...")
    print(f"{'=' * 56}\n")

    async def _research_task():
        """Run research pipeline with pre-gathered inputs."""
        _log("Starting research...\n")
        result = await deep_research(
            query=research_combined_query,
            breadth=breadth,
            depth=depth,
        )
        _log("\nResearch complete.")
        _log("Writing final output...")

        outputs_dir = run_dir
        outputs_dir.mkdir(parents=True, exist_ok=True)

        if is_report:
            report = write_final_report(
                prompt=research_combined_query,
                learnings=result.learnings,
                visited_urls=result.visited_urls,
                relevant_images=result.relevant_images,
            )
            report_path = outputs_dir / "report.md"
            report_path.write_text(report, encoding="utf-8")
            _log(f"\n[Research] Report saved to {report_path}")
            return {"type": "report", "path": str(report_path), "learnings": len(result.learnings), "urls": len(result.visited_urls)}
        else:
            answer = write_final_answer(
                prompt=research_combined_query,
                learnings=result.learnings,
            )
            answer_path = outputs_dir / "answer.md"
            answer_path.write_text(answer, encoding="utf-8")
            _log(f"\n[Research] Answer saved to {answer_path}")
            return {"type": "answer", "path": str(answer_path), "learnings": len(result.learnings), "urls": len(result.visited_urls)}

    def _cloud_task():
        """Run cloud pipeline with pre-gathered inputs (unattended)."""
        return _cloud_continue_flow(
            user_requirement=query,
            selected_framework=selected_framework,
            selected_accelerator=selected_accelerator,
            selected_dataset=selected_dataset,
            selected_vm=selected_vm,
            questions_payload=questions_payload,
            answers=answers,
            has_user_script=has_user_script,
            user_script_content=user_script_content,
            # run_now / keep_vm_choice intentionally omitted so the cloud flow
            # prompts "run training now?" / "keep the Droplet?" interactively
            # at execution time. non_interactive=True only auto-cancels the
            # rare dataset-inspection-failure fallback (its default is no).
            non_interactive=True,
            run_dir=run_dir,
        )

    # Run both concurrently — research is native async, cloud is sync via to_thread
    research_task = asyncio.create_task(_research_task())
    cloud_task = asyncio.create_task(asyncio.to_thread(_cloud_task))

    research_result, cloud_result = await asyncio.gather(
        research_task, cloud_task, return_exceptions=True
    )

    print(f"\n{'=' * 56}")
    print(f"  Results Summary")
    print(f"{'=' * 56}")

    if isinstance(research_result, BaseException):
        print(f"\n[Research Agent] FAILED: {research_result}")
        research_result = {"status": "error", "error": str(research_result)}
    else:
        print(f"\n[Research Agent] Completed.")
        if research_result.get("type") == "report":
            print(f"  Report: {research_result.get('path')}")
        else:
            print(f"  Answer: {research_result.get('path')}")
        print(f"  Learnings: {research_result.get('learnings', 0)}")
        print(f"  URLs visited: {research_result.get('urls', 0)}")

    if isinstance(cloud_result, BaseException):
        print(f"\n[Cloud Experiment Agent] FAILED: {cloud_result}")
        cloud_result = {"status": "error", "error": str(cloud_result)}
    else:
        status = cloud_result.get("status", "unknown")
        print(f"\n[Cloud Experiment Agent] Status: {status}")
        if status == "completed":
            print(f"  Job ID: {cloud_result.get('job_id')}")
            print(f"  Workspace: {cloud_result.get('workspace')}")

    # ── Combined final report (only when the cloud task completed) ──────
    # F2: previously this ran _generate_final_report("") when the cloud task
    # failed, resolving to a bogus outputs/outputs path and producing a
    # research-only report without flagging the cloud failure. Now we only
    # synthesize the combined report when the cloud agent actually completed,
    # and surface the failure clearly otherwise.
    print()
    cloud_status = cloud_result.get("status") if isinstance(cloud_result, dict) else "error"
    final_report_status = "skipped"
    if cloud_status == "completed":
        cloud_job_id = cloud_result.get("job_id", "")
        final_report_status = _generate_final_report(cloud_job_id, run_dir=run_dir)
    else:
        print("[Final Report] Skipping combined final report — the Cloud Experiment Agent did not complete.")
        if isinstance(research_result, dict) and research_result.get("path"):
            print(f"  Research output is available at: {research_result.get('path')}")
        if isinstance(cloud_result, dict) and cloud_result.get("workspace"):
            print(f"  Prepared cloud files: {cloud_result.get('workspace')}")

    return {
        "research": research_result,
        "cloud": cloud_result,
        "final_report_status": final_report_status,
        "run_dir": str(run_dir),
    }




# ---------------------------------------------------------------------------
# 8. Agent Selection Menu
# ---------------------------------------------------------------------------


def _show_menu():
    print(f"\n{'=' * 56}")
    print(f"  Orchestrator — Agent Selection")
    print(f"{'=' * 56}")
    print(f"  1. Research Agent")
    print(f"     Deep web research, question feedback, Markdown report generation")
    print(f"  2. Cloud Experiment Agent")
    print(f"     ML job configuration, AMD VM infrastructure, remote execution")
    print(f"  3. Both Workflows")
    print(f"     Sequential questioning → parallel execution")
    print(f"{'=' * 56}")
    while True:
        choice = input("\nEnter your choice (1, 2, or 3): ").strip()
        if choice in ("1", "2", "3"):
            return int(choice)
        print("Please enter 1, 2, or 3.")


def _generate_report_standalone() -> str:
    """Standalone: detect available run folders and (re)generate a final report.

    Scans ``final_amd/outputs/`` for per-execution run folders (the new layout):
    a folder qualifies if it has an ``outputs/`` subdir (cloud experiment
    artifacts) **or** a ``runtime_generated/job_spec.json`` (prepared payload).

    Returns the final-report status (``ok``/``skipped``/``failed``) so the
    caller can reflect it in the exit banner / return code (F3).
    """
    outputs_root = _BASE_DIR / "outputs"
    outputs_root.mkdir(parents=True, exist_ok=True)

    # ── Detect available run folders ────────────────────────────
    run_folders: list[Path] = []
    for entry in sorted(outputs_root.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        has_outputs = (entry / "outputs").is_dir()
        has_payload = (entry / "runtime_generated" / "job_spec.json").exists()
        if has_outputs or has_payload:
            run_folders.append(entry)

    if not run_folders:
        print("\nNo cloud experiment run folders found.")
        print(f"  Checked: {outputs_root.relative_to(_BASE_DIR)}/")
        print("Run the Cloud Experiment (option 2) first to produce a run folder.")
        return "skipped"

    # ── Pick run folder ─────────────────────────────────────────
    print(f"\nDetected cloud experiment run folder(s):")
    for i, folder in enumerate(run_folders, 1):
        print(f"  [{i}] {folder.name}")
    print(f"  [0] Cancel")

    while True:
        raw = input("\nSelect a run folder: ").strip()
        if raw == "0":
            print("Cancelled.")
            return "skipped"
        if raw.isdigit() and 1 <= int(raw) <= len(run_folders):
            selected = run_folders[int(raw) - 1]
            break
        print(f"Enter a number between 0 and {len(run_folders)}.")

    # ── Generate report inside the selected run folder ──────────
    # job_id is read from the prepared job_spec if present (cosmetic — only
    # used in log lines), else the folder name.
    job_spec_path = selected / "runtime_generated" / "job_spec.json"
    if job_spec_path.exists():
        try:
            job_id = json.loads(job_spec_path.read_text(encoding="utf-8")).get("job_id", selected.name)
        except (json.JSONDecodeError, OSError):
            job_id = selected.name
    else:
        job_id = selected.name
    print()
    return _generate_final_report(str(job_id), run_dir=selected)


# ---------------------------------------------------------------------------
# 9. Main Entry Point
# ---------------------------------------------------------------------------


def _print_exit_banner(report_status: str) -> None:
    """Print the closing banner, reflecting final-report status (F3).

    Previously every path printed a plain "Done." even when the capstone
    final-report LLM call had failed, leaving the user with no signal that the
    report was missing. Now a failed report produces a warning banner.
    """
    print(f"\n{'=' * 56}")
    if report_status == "failed":
        print(f"  Done with warnings — final report generation FAILED.")
        print(f"  See [Final Report] messages above; other artifacts were still saved.")
    else:
        print(f"  Done.")
    print(f"{'=' * 56}\n")


async def main() -> int:
    parser = argparse.ArgumentParser(description="Unified Orchestrator — Deep Research + Cloud Experiment")
    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="Read an existing run folder's report.md and cloud artifacts to (re)generate final-report.md inside it",
    )
    parser.add_argument(
        "--gen-report",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args, _ = parser.parse_known_args()

    if args.generate_report or args.gen_report:
        report_status = _generate_report_standalone()
        _print_exit_banner(report_status)
        return 1 if report_status == "failed" else 0

    choice = _show_menu()
    report_status = "skipped"

    # Single unified query
    if choice == 1:
        await _research_interactive()
    elif choice == 2:
        result = _cloud_interactive()
        if isinstance(result, dict) and result.get("status") == "completed":
            _rd = result.get("run_dir")
            if _rd:
                print()
                report_status = _generate_final_report(
                    result.get("job_id", ""),
                    run_dir=Path(_rd),
                )
    elif choice == 3:
        both = await _run_both_workflows()
        if isinstance(both, dict):
            report_status = both.get("final_report_status", "skipped")

    _print_exit_banner(report_status)
    return 1 if report_status == "failed" else 0


if __name__ == "__main__":
    try:
        _exit_code = asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
    sys.exit(_exit_code)
