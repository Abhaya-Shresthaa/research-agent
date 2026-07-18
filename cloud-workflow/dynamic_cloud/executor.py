from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from dynamic_cloud.amd_droplet import AmdDropletManager
from dynamic_cloud.config import AmdDropletSession, AmdSettings
from dynamic_cloud.docker_runner import RemoteHostRunner
from dynamic_cloud.workspace import JobWorkspace

try:
    from src.ai.providers import get_client as _report_llm_client, get_model_id as _report_llm_model
    _HAS_REPORT_LLM = True
except ImportError:
    _HAS_REPORT_LLM = False


def _extract_sources_from_research(report_path: str) -> list[str]:
    """Extract URLs from the Sources section of the research report.md."""
    try:
        text = Path(report_path).read_text(encoding="utf-8")
    except (OSError, ValueError):
        return []
    lines = text.splitlines()
    in_sources = False
    urls: list[str] = []
    for line in lines:
        if line.strip().startswith("## Sources"):
            in_sources = True
            continue
        if in_sources:
            if line.startswith("## "):
                break
            stripped = line.strip().strip("-").strip()
            if stripped.startswith("http://") or stripped.startswith("https://"):
                urls.append(stripped)
    return urls


def _generate_final_report(job_id: str, run_dir: Path) -> str:
    """Synthesize a comprehensive final-report.md from research + experiment outputs.

    Mirrors the ``write_final_report`` pattern from the research-workflow:
    1. Extract metadata/images as structured XML for the LLM
    2. Call LLM with response_format=json_object
    3. Post-check what images/plots the LLM embedded inline
    4. Append any missing ones at the bottom with context
    5. Append sources section from research report

    All artifacts are read from / written into the per-execution folder
    ``run_dir`` (``outputs/<runtime_name>``):
      - research report:  ``<run_dir>/report.md``
      - cloud outputs:    ``<run_dir>/outputs/``
      - this report:      ``<run_dir>/final-report.md``

    Returns a status string so callers can reflect the outcome in their exit
    banner / return code:
      - ``"ok"``      : report written
      - ``"skipped"`` : nothing to report (no LLM, no data, or empty LLM reply)
      - ``"failed"``  : the LLM call raised and no report was written
    """
    if not _HAS_REPORT_LLM:
        print("  [Final Report] LLM client not available (research module not loaded) — skipping.")
        return "skipped"

    sections: list[dict[str, str]] = []

    research_path = run_dir / "report.md"
    outputs_root = run_dir / "outputs"
    final_path = run_dir / "final-report.md"

    # ── 1. Research report (if it exists from a parallel research run) ──
    has_research = research_path.exists()
    if has_research:
        report_text = research_path.read_text(encoding="utf-8")
        sections.append({
            "source": "Deep Web Research Report",
            "content": report_text,
        })
        print("  [Final Report] Loaded research report.")

    # ── 2. Cloud experiment outputs ────────────────────────────────────
    if outputs_root.exists():
        # environment.json
        env_path = outputs_root / "environment.json"
        if env_path.exists():
            env_data = json.loads(env_path.read_text(encoding="utf-8"))
            sections.append({
                "source": "Experiment Environment",
                "content": json.dumps({
                    "python_version": env_data.get("python_version"),
                    "packages": env_data.get("packages", {}),
                    "hyperparameters": env_data.get("hyperparameters", {}),
                    "dataset_info": env_data.get("dataset", {}),
                    "hardware": env_data.get("hardware", {}),
                }, indent=2),
            })

        # metrics.json
        metrics_path = outputs_root / "metrics.json"
        if metrics_path.exists():
            sections.append({
                "source": "Experiment Metrics",
                "content": json.dumps(json.loads(metrics_path.read_text(encoding="utf-8")), indent=2),
            })

        # training_logs.json
        logs_path = outputs_root / "training_logs.json"
        if logs_path.exists():
            sections.append({
                "source": "Training Logs (per-epoch)",
                "content": json.dumps(json.loads(logs_path.read_text(encoding="utf-8")), indent=2),
            })

        # runtime.log
        runtime_log_path = outputs_root / "logs" / "runtime.log"
        if runtime_log_path.exists():
            sections.append({
                "source": "Runtime Log",
                "content": runtime_log_path.read_text(encoding="utf-8"),
            })

        # output_report.txt
        report_txt_path = outputs_root / "logs" / "output_report.txt"
        if report_txt_path.exists():
            sections.append({
                "source": "Output Report",
                "content": report_txt_path.read_text(encoding="utf-8"),
            })

        # dataset_metadata.json
        dsm_path = outputs_root / "dataset_metadata.json"
        if dsm_path.exists():
            sections.append({
                "source": "Dataset Metadata",
                "content": json.dumps(json.loads(dsm_path.read_text(encoding="utf-8")), indent=2)[:8000],
            })

        print(f"  [Final Report] Loaded cloud experiment outputs ({job_id}).")
    else:
        print(f"  [Final Report] Cloud outputs not found at {outputs_root}.")

    if not sections:
        print("  [Final Report] No data available — skipping.")
        return "skipped"

    # ── 3a. Extract research report images (web URLs) ────────────────
    # Pattern from research-workflow: structured <image> XML blocks
    research_images: list[dict[str, str]] = []
    if has_research:
        report_text = research_path.read_text(encoding="utf-8")
        img_pattern = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
        for match in img_pattern.finditer(report_text):
            alt = match.group(1).strip()
            url = match.group(2).strip()
            if url and not url.startswith("data:"):
                # Try to grab surrounding context (~300 chars around the image)
                start = max(0, match.start() - 300)
                end = min(len(report_text), match.end() + 300)
                context = re.sub(r"\s+", " ", report_text[start:end]).strip()
                research_images.append({
                    "image_url": url,
                    "alt_text": alt or "Research image",
                    "source_url": url,
                    "context": context[:500],
                    "relevance": f"Illustrates {alt or 'a concept from web research'}",
                })
                if len(research_images) >= 6:
                    break

    # ── 3b. Detect experiment plots (local files) ────────────────────
    # We treat these like images but use local relative paths as the URL.
    # plots_root / metrics_path_ck are derived from outputs_root (the run
    # folder's outputs/ directory).
    experiment_plots: list[dict[str, str]] = []
    plots_root = outputs_root / "plots"
    metrics_data: dict[str, Any] = {}
    metrics_path_ck = outputs_root / "metrics.json"
    if metrics_path_ck.exists():
        try:
            metrics_data = json.loads(metrics_path_ck.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            metrics_data = {}
    if plots_root.exists():
        # final-report.md lives at final_path inside the run folder, so image
        # links must be relative to final_path.parent (the run folder).
        report_dir = final_path.parent
        for pf in sorted(plots_root.iterdir()):
            rel_path = os.path.relpath(pf, report_dir)
            # Build context from available metrics
            context_parts = []
            if "final_train_accuracy" in metrics_data:
                context_parts.append(f"Final train accuracy: {metrics_data['final_train_accuracy']:.4f}")
            if "final_val_accuracy" in metrics_data:
                context_parts.append(f"Final validation accuracy: {metrics_data['final_val_accuracy']:.4f}")
            if "final_train_loss" in metrics_data:
                context_parts.append(f"Final train loss: {metrics_data['final_train_loss']:.4f}")
            if "test_accuracy" in metrics_data:
                context_parts.append(f"Test accuracy: {metrics_data['test_accuracy']:.4f}")
            ctx = "; ".join(context_parts) if context_parts else "Training history plot"
            experiment_plots.append({
                "image_url": rel_path,
                "alt_text": pf.stem.replace("_", " ").replace("-", " ").title(),
                "source_url": rel_path,
                "context": ctx,
                "relevance": f"Experiment training plot: {pf.stem}",
            })

    # ── 3c. Build structured XML (same format as research-workflow) ──
    sections_xml = "\n\n".join(
        f'<section source="{s["source"]}">\n{s["content"]}\n</section>'
        for s in sections
    )

    research_images_xml = ""
    if research_images:
        research_images_xml = "<research_images>\n" + "\n".join(
            (
                "<image>\n"
                f"<image_url>{img['image_url']}</image_url>\n"
                f"<alt_text>{img['alt_text']}</alt_text>\n"
                f"<source_url>{img['source_url']}</source_url>\n"
                f"<context>{img['context']}</context>\n"
                f"<relevance>{img['relevance']}</relevance>\n"
                "</image>"
            )
            for img in research_images
        ) + "\n</research_images>"

    plots_xml = ""
    if experiment_plots:
        plots_xml = "<experiment_plots>\n" + "\n".join(
            (
                "<image>\n"
                f"<image_url>{p['image_url']}</image_url>\n"
                f"<alt_text>{p['alt_text']}</alt_text>\n"
                f"<source_url>{p['source_url']}</source_url>\n"
                f"<context>{p['context']}</context>\n"
                f"<relevance>{p['relevance']}</relevance>\n"
                "</image>"
            )
            for p in experiment_plots
        ) + "\n</experiment_plots>"

    if has_research:
        prompt = (
            "You are a senior ML research scientist writing a comprehensive unified report. "
            "You have been given two independent streams of information:\n\n"
            "1. A **Deep Web Research Report** — findings from web searches (papers, articles, "
            "documentation) about a topic.\n"
            "2. A **Cloud Experiment Results** — empirical outputs from an actual ML training "
            "run on AMD GPU (metrics, hyperparameters, environment, training curves, logs).\n\n"
            "Synthesize a detailed, well-structured markdown report that:\n\n"
            "- Starts with an **Executive Summary** that introduces the topic and the dual "
            "nature of the investigation (theoretical/web research + hands-on experiment).\n"
            "- Presents the **Web Research Findings** in a clear, condensed form — key "
            "concepts, architectures, techniques, comparisons, and sources.\n"
            "- Presents the **Experiment Results** — what was trained, on what hardware, "
            "with what hyperparameters, and what metrics were achieved. Embed experiment "
            "plots in this section using the paths provided.\n"
            "- **Compares** the research expectations/theory with the actual empirical results. "
            "Where does theory align with practice? Where are there gaps or contradictions?\n"
            "- Highlights **Key Insights & Takeaways** — what worked, what didn't, surprising "
            "findings, and practical lessons.\n"
            "- Includes **All Sources & References** — URLs from web research, dataset sources, "
            "libraries/frameworks used.\n\n"
            "Be thorough and analytical. Embed concrete numbers (accuracy, loss, training time, "
            "hardware specs) from the experiment section. Connect web-researched concepts to "
            "their observed manifestations (or lack thereof) in the experiment.\n\n"
            f"{research_images_xml}\n\n"
            f"{plots_xml}\n\n"
            "Embed the most important research images and experiment plots directly in the "
            "report body using markdown syntax. "
            "For web research images from <research_images>: use ![alt text](image_url) "
            "where the concepts they illustrate are discussed. "
            "For experiment plots from <experiment_plots>: use ![alt text](image_url) "
            "in the Experiment Results section where training performance is discussed. "
            "Only include images that genuinely support the surrounding text — "
            "skip decorative or marginal ones.\n\n"
            "Here is all the data:\n\n"
            f"{sections_xml}\n\n"
            "Respond with valid JSON only: {\"report_markdown\": \"...\"}"
        )
    else:
        prompt = (
            "You are a senior ML research scientist writing a detailed experiment report. "
            "Given the following cloud ML experiment outputs, write a comprehensive markdown "
            "report that covers:\n\n"
            "- **Executive Summary** — what was trained, goal of the experiment\n"
            "- **Experiment Setup** — hardware (AMD GPU), hyperparameters, dataset, framework\n"
            "- **Training Results** — metrics, accuracy, loss, training time, per-epoch progression. "
            "Embed experiment plots in this section using the paths provided.\n"
            "- **Analysis** — interpretation of the results, what worked well, areas for improvement\n"
            "- **Key Takeaways** — practical lessons from this run\n\n"
            "Be thorough and analytical. Embed concrete numbers and comparisons.\n\n"
            f"{plots_xml}\n\n"
            "Embed the experiment plots directly in the Experiment Results section "
            "using markdown syntax: ![alt text](image_url). "
            "Always include the training history plot if available.\n\n"
            "Here is all the data:\n\n"
            f"{sections_xml}\n\n"
            "Respond with valid JSON only: {\"report_markdown\": \"...\"}"
        )

    # ── 4. Call the LLM ───────────────────────────────────────────────
    print("  [Final Report] Generating unified report via LLM...")
    try:
        client = _report_llm_client()
        model_id = _report_llm_model()
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert ML research analyst. You write thorough, "
                        "well-organized markdown reports that integrate theoretical research "
                        "with empirical experimental results. Always respond with valid JSON "
                        "containing a single key 'report_markdown' whose value is the full "
                        "report in GitHub-flavored markdown."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        reply = json.loads(content)
        report_markdown = reply.get("report_markdown", "")

        if not report_markdown.strip():
            print("  [Final Report] LLM returned empty report — skipping.")
            return "skipped"

        # ── 5. Post-check: only force-append experiment plots ────────────
        # (research images are passed for inline embedding but never dumped at the bottom)
        extra_sections: list[str] = []

        # 5a. Only experiment plots that the LLM didn't embed inline
        missing_plots = [
            p for p in experiment_plots
            if p["image_url"] not in report_markdown
        ]
        if missing_plots:
            sec = "\n\n## Experiment Plots\n\n" + "\n\n".join(
                (
                    f"![{p['alt_text']}]({p['image_url']})\n\n"
                    f"*{p['context']}*"
                )
                for p in missing_plots[:3]
            )
            extra_sections.append(sec)

        # 5b. Sources section from research report
        urls_section = ""
        if has_research:
            visited = _extract_sources_from_research(str(research_path))
            if visited:
                urls_section = "\n\n## Sources & References\n\n" + "\n".join(f"- {url}" for url in visited)

        # ── 6. Write the file ─────────────────────────────────────────
        final_markdown = report_markdown + "".join(extra_sections) + urls_section

        # final_path is <run_dir>/final-report.md.
        final_path.parent.mkdir(parents=True, exist_ok=True)
        final_path.write_text(final_markdown, encoding="utf-8")
        embedded_plots = len(experiment_plots) - len(missing_plots)
        print(f"  [Final Report] Saved to {final_path}")
        if research_images:
            print(f"    Research images: {len(research_images)} passed to LLM for inline embedding")
        print(f"    Experiment plots embedded inline: {embedded_plots}/{len(experiment_plots)}"
              f"{' (+ ' + str(len(missing_plots)) + ' appended at bottom)' if missing_plots else ''}")
        return "ok"

    except BaseException as exc:
        print(f"  [Final Report] Generation failed: {exc}")
        return "failed"


def execute_workspace_on_amd(
    workspace: JobWorkspace,
    job_spec: dict[str, Any],
    amd_settings: AmdSettings,
    keep_vm: bool = False,
    preserve_remote: bool = False,
    vm_already_selected: bool = False,
    session: AmdDropletSession | None = None,
) -> None:
    """Execute a prepared workspace on an AMD GPU Droplet.

    Lifecycle: create/adopt → cloud-init/local-image readiness → upload → run
    containers → download → destroy. The AMD cloud-init payload builds the
    runtime image locally; execution never pulls a Docker image.

    Outputs are downloaded directly into ``workspace.outputs_dir`` (the
    per-execution run folder's ``outputs/``), so there is no post-run shift.
    """
    if vm_already_selected and not session:
        raise RuntimeError(
            "vm_already_selected=True requires an AmdDropletSession with droplet_id and ip."
        )

    droplet = AmdDropletManager(amd_settings)
    host = RemoteHostRunner(amd_settings)

    run_error: BaseException | None = None
    local_outputs: Path | None = None
    try:
        if session:
            droplet.adopt(session.droplet_id)
            ip = session.ip
            print(f"  ▸ Reusing existing Droplet {session.droplet_id}")
            droplet.wait_for_ssh(ip)
        else:
            droplet.create_droplet()
            droplet.wait_until_active()
            ip = droplet.get_public_ip()
        print(f"  ✓ Droplet ready at {ip}")

        # Upload payload and execute it in the cloud-init-built local image.
        remote_job_dir = host.upload_workspace(ip, workspace, preserve_remote=preserve_remote)
        host.execute_bootstrap(ip, remote_job_dir)
        local_outputs = host.download_outputs(ip, workspace, remote_job_dir)
        print(f"\nTraining finished. Outputs downloaded to: {local_outputs}")
    except BaseException as exc:
        run_error = exc
        raise
    finally:
        if keep_vm:
            print("Keeping Droplet because you selected that option.")
            print(f"IMPORTANT: Destroy it manually to stop billing!")
            if droplet.droplet_id:
                print(
                    f"  Destroy: curl -X DELETE -H 'Authorization: Bearer $AMD_TOKEN' "
                    f"{droplet.base_url}/droplets/{droplet.droplet_id}"
                )
        else:
            # Shield destroy from KeyboardInterrupt — a second Ctrl-C MUST NOT
            # leak the droplet.  We retry destroy indefinitely until it succeeds
            # or the process is forcibly killed (SIGKILL).
            destroy_attempts = 0
            while True:
                destroy_attempts += 1
                try:
                    droplet.destroy()
                    break  # destroy succeeded
                except KeyboardInterrupt:
                    print(
                        f"\n  Destroy in progress — this stops billing for "
                        f"Droplet {droplet.droplet_id}. Please wait..."
                    )
                    # Continue the loop — never let Ctrl-C abort destroy
                except BaseException as cleanup_exc:
                    print(f"Cleanup failed: {cleanup_exc}")
                    if droplet.droplet_id:
                        print(
                            f"IMPORTANT: Manually destroy Droplet {droplet.droplet_id} to stop billing!\n"
                            f"  Destroy: curl -X DELETE -H 'Authorization: Bearer $AMD_TOKEN' "
                            f"{droplet.base_url}/droplets/{droplet.droplet_id}"
                        )
                    if run_error is None:
                        raise
                    print("The original run error above is the important one.")
                    break
