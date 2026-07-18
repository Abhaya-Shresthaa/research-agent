from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from dynamic_cloud.config import AmdDropletSession, AmdSettings
from dynamic_cloud.executor import execute_workspace_on_amd
from dynamic_cloud.workspace import JobWorkspace


class RecordingDroplet:
    instances: list["RecordingDroplet"] = []

    def __init__(self, settings: AmdSettings) -> None:
        self.settings = settings
        self.droplet_id: int | None = None
        self.created = False
        self.destroyed = False
        RecordingDroplet.instances.append(self)

    def adopt(self, droplet_id: int) -> None:
        self.droplet_id = droplet_id

    def create_droplet(self) -> None:
        self.created = True
        self.droplet_id = 456

    def wait_until_active(self) -> None:
        pass

    def wait_for_ssh(self, ip: str) -> None:
        pass

    def get_public_ip(self) -> str:
        return "198.51.100.12"

    def destroy(self) -> None:
        self.destroyed = True
        self.droplet_id = None


class RecordingHost:
    instances: list["RecordingHost"] = []

    def __init__(self, settings: AmdSettings) -> None:
        self.settings = settings
        self.uploads: list[tuple[str, bool]] = []
        RecordingHost.instances.append(self)

    def upload_workspace(self, ip: str, workspace: JobWorkspace, preserve_remote: bool = False) -> str:
        self.uploads.append((ip, preserve_remote))
        return "/root/dynamic_jobs/job"

    def execute_bootstrap(self, ip: str, remote_job_dir: str) -> None:
        pass

    def download_outputs(self, ip: str, workspace: JobWorkspace, remote_job_dir: str) -> Path:
        return workspace.root / "outputs"


class ExecutorSessionTests(unittest.TestCase):
    def setUp(self) -> None:
        RecordingDroplet.instances = []
        RecordingHost.instances = []

    def test_selected_vm_requires_session(self) -> None:
        workspace = self._workspace()
        with self.assertRaisesRegex(RuntimeError, "AmdDropletSession"):
            execute_workspace_on_amd(
                workspace,
                {"runtime": {"image": "image:latest", "accelerator": "gpu"}},
                self._settings(),
                vm_already_selected=True,
            )

    def test_session_reuses_droplet_and_destroys_when_done(self) -> None:
        workspace = self._workspace()
        session = AmdDropletSession(
            droplet_id=123,
            ip="203.0.113.10",
            vm_name="inspection-vm",
            remote_job_dir="/root/dynamic_jobs/inspect",
            created_for_dataset_inspection=True,
        )

        with mock.patch("dynamic_cloud.executor.AmdDropletManager", RecordingDroplet), mock.patch(
            "dynamic_cloud.executor.RemoteHostRunner", RecordingHost
        ):
            execute_workspace_on_amd(
                workspace,
                {"runtime": {"image": "image:latest", "accelerator": "gpu"}},
                self._settings(),
                preserve_remote=True,
                vm_already_selected=True,
                session=session,
            )

        droplet = RecordingDroplet.instances[0]
        host = RecordingHost.instances[0]
        self.assertFalse(droplet.created)
        self.assertTrue(droplet.destroyed)
        self.assertEqual(host.uploads, [("203.0.113.10", True)])

    # ── D1: destroy shield — a second Ctrl-C must NOT leak the droplet ────

    def test_destroy_retried_after_keyboard_interrupt(self) -> None:
        class InterruptOnceDroplet(RecordingDroplet):
            destroy_calls = 0

            def destroy(self) -> None:
                InterruptOnceDroplet.destroy_calls += 1
                if InterruptOnceDroplet.destroy_calls == 1:
                    raise KeyboardInterrupt
                super().destroy()

        workspace = self._workspace()
        session = AmdDropletSession(
            droplet_id=777,
            ip="203.0.113.10",
            vm_name="inspection-vm",
            remote_job_dir="/root/dynamic_jobs/inspect",
            created_for_dataset_inspection=True,
        )

        with mock.patch("dynamic_cloud.executor.AmdDropletManager", InterruptOnceDroplet), mock.patch(
            "dynamic_cloud.executor.RemoteHostRunner", RecordingHost
        ):
            execute_workspace_on_amd(
                workspace,
                {"runtime": {"image": "image:latest", "accelerator": "gpu"}},
                self._settings(),
                vm_already_selected=True,
                session=session,
            )

        droplet = InterruptOnceDroplet.instances[0]
        # destroy() was called twice (first interrupted, second succeeded) and
        # the droplet ended up destroyed — no KeyboardInterrupt leaked out.
        self.assertEqual(InterruptOnceDroplet.destroy_calls, 2)
        self.assertTrue(droplet.destroyed)

    def _workspace(self) -> JobWorkspace:
        root = Path(tempfile.mkdtemp())
        payload = root / "payload"
        generated = payload / "generated"
        data = payload / "data"
        generated.mkdir(parents=True)
        data.mkdir()
        (payload / "metadata.json").write_text("{}", encoding="utf-8")
        return JobWorkspace("job", root, payload, generated, data, root / "outputs")

    def _settings(self) -> AmdSettings:
        return AmdSettings(api_key="token", ssh_private_key_path=Path("/tmp/key"))


# ═══════════════════════════════════════════════════════════════════════════
# Section F — _generate_final_report return-status tests (F3)
# ═══════════════════════════════════════════════════════════════════════════


class FinalReportStatusTests(unittest.TestCase):
    """Verify that _generate_final_report returns the correct status strings
    ("ok" / "skipped" / "failed") for each code path, so callers can reflect
    the outcome in their exit banner and return code (Section F, F3)."""

    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.temp = Path(self._temp_dir.name)

        # _generate_final_report reads/writes everything inside run_dir.
        (self.temp / "outputs").mkdir(parents=True, exist_ok=True)
        self.run_dir = self.temp / "outputs" / "test-run"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # By default, mock the LLM as unavailable (tests the "skipped" path).
        self._has_llm_patch = mock.patch("dynamic_cloud.executor._HAS_REPORT_LLM", False)
        self._has_llm_patch.start()

    def tearDown(self) -> None:
        self._has_llm_patch.stop()
        self._temp_dir.cleanup()

    # ── Helper ──────────────────────────────────────────────────────────

    def _make_cloud_outputs(self, job_id: str) -> Path:
        """Create a minimal set of cloud outputs inside the run folder."""
        outputs_root = self.run_dir / "outputs"
        outputs_root.mkdir(parents=True, exist_ok=True)
        (outputs_root / "metrics.json").write_text(
            json.dumps({"final_train_accuracy": 0.95, "final_val_accuracy": 0.88}),
            encoding="utf-8",
        )
        (outputs_root / "environment.json").write_text(
            json.dumps({"python_version": "3.11", "framework": "pytorch"}),
            encoding="utf-8",
        )
        return outputs_root

    @staticmethod
    def _enable_llm_on_module() -> None:
        """Directly set _HAS_REPORT_LLM=True and inject the missing LLM helpers
        on the executor module so _generate_final_report can call them."""
        import dynamic_cloud.executor as _exc
        _exc._HAS_REPORT_LLM = True
        if not hasattr(_exc, "_report_llm_client"):
            _exc._report_llm_client = mock.MagicMock()
        if not hasattr(_exc, "_report_llm_model"):
            _exc._report_llm_model = lambda: "test-model"

    @staticmethod
    def _disable_llm_on_module() -> None:
        """Restore the module state so the LLM is unavailable again."""
        import dynamic_cloud.executor as _exc
        _exc._HAS_REPORT_LLM = False
        # Remove injected helpers so other tests start clean.
        for _name in ("_report_llm_client", "_report_llm_model"):
            if hasattr(_exc, _name):
                delattr(_exc, _name)

    # ── Tests ───────────────────────────────────────────────────────────

    def test_returns_skipped_when_llm_not_available(self) -> None:
        from dynamic_cloud.executor import _generate_final_report

        self._make_cloud_outputs("test-job")
        result = _generate_final_report("test-job", run_dir=self.run_dir)
        self.assertEqual(result, "skipped")

    def test_returns_skipped_when_no_data_at_all(self) -> None:
        from dynamic_cloud.executor import _generate_final_report

        # run_dir exists but has no outputs and no research report.
        result = _generate_final_report("nonexistent-job", run_dir=self.run_dir)
        self.assertEqual(result, "skipped")

    def test_returns_skipped_when_llm_returns_empty_report(self) -> None:
        from dynamic_cloud.executor import _generate_final_report

        self._enable_llm_on_module()
        self.addCleanup(self._disable_llm_on_module)

        import dynamic_cloud.executor as _exc
        fake_client = mock.MagicMock()
        fake_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"report_markdown": ""}'))]
        )

        # _report_llm_client is called as `client = _report_llm_client()`, so
        # we must patch the function, not the client instance itself.
        with mock.patch.object(_exc, "_report_llm_client", return_value=fake_client, create=True), \
             mock.patch.object(_exc, "_report_llm_model", return_value="test-model"):
            self._make_cloud_outputs("test-job")
            result = _generate_final_report("test-job", run_dir=self.run_dir)
            self.assertEqual(result, "skipped")

    def test_returns_ok_when_report_written_successfully(self) -> None:
        from dynamic_cloud.executor import _generate_final_report

        self._enable_llm_on_module()
        self.addCleanup(self._disable_llm_on_module)

        import dynamic_cloud.executor as _exc
        report_content = "# Unified Report\n\nExperiment results here."
        fake_client = mock.MagicMock()
        fake_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content=json.dumps({"report_markdown": report_content})
            ))]
        )

        with mock.patch.object(_exc, "_report_llm_client", return_value=fake_client, create=True), \
             mock.patch.object(_exc, "_report_llm_model", return_value="test-model"):
            self._make_cloud_outputs("test-job")
            result = _generate_final_report("test-job", run_dir=self.run_dir)
            self.assertEqual(result, "ok")

        # The report file was written inside the run folder.
        report_path = self.run_dir / "final-report.md"
        self.assertTrue(report_path.exists(), f"Expected report at {report_path}")
        written = report_path.read_text(encoding="utf-8")
        self.assertIn(report_content, written)

    def test_returns_failed_when_llm_raises(self) -> None:
        from dynamic_cloud.executor import _generate_final_report

        self._enable_llm_on_module()
        self.addCleanup(self._disable_llm_on_module)

        import dynamic_cloud.executor as _exc
        fake_client = mock.MagicMock()
        fake_client.chat.completions.create.side_effect = RuntimeError("API error")

        with mock.patch.object(_exc, "_report_llm_client", return_value=fake_client, create=True), \
             mock.patch.object(_exc, "_report_llm_model", return_value="test-model"):
            self._make_cloud_outputs("test-job")
            result = _generate_final_report("test-job", run_dir=self.run_dir)
            self.assertEqual(result, "failed")

        # The report file must NOT exist when generation failed.
        report_path = self.run_dir / "final-report.md"
        self.assertFalse(report_path.exists(), f"Report should not exist at {report_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Section F — _print_exit_banner tests (F3)
# ═══════════════════════════════════════════════════════════════════════════


class ExitBannerTests(unittest.TestCase):
    """Verify _print_exit_banner reflects final-report status (F3).

    _print_exit_banner is defined in main.py, which does import-time side
    effects (env loading, sys.path manipulation, monkey-patching) that are
    impractical to mock in a unit test.  The function is pure I/O — it only
    calls print() — so we mirror its exact source here.  If main.py ever
    changes the banner format, this test must be updated to match.

    Source mirrors main.py:_print_exit_banner (lines 787-800).
    """

    @staticmethod
    def _print_exit_banner(report_status: str) -> None:
        print(f"\n{'=' * 56}")
        if report_status == "failed":
            print(f"  Done with warnings — final report generation FAILED.")
            print(f"  See [Final Report] messages above; other artifacts were still saved.")
        else:
            print(f"  Done.")
        print(f"{'=' * 56}\n")

    def test_ok_status_prints_done(self) -> None:
        captured = io.StringIO()
        with mock.patch("sys.stdout", captured):
            self._print_exit_banner("ok")

        output = captured.getvalue()
        self.assertIn("Done.", output)
        self.assertNotIn("FAILED", output)

    def test_skipped_status_prints_done(self) -> None:
        captured = io.StringIO()
        with mock.patch("sys.stdout", captured):
            self._print_exit_banner("skipped")

        output = captured.getvalue()
        self.assertIn("Done.", output)
        self.assertNotIn("FAILED", output)

    def test_failed_status_prints_warning(self) -> None:
        captured = io.StringIO()
        with mock.patch("sys.stdout", captured):
            self._print_exit_banner("failed")

        output = captured.getvalue()
        self.assertIn("Done with warnings", output)
        self.assertIn("FAILED", output)
        self.assertIn("[Final Report]", output)


# ═══════════════════════════════════════════════════════════════════════════
# Per-execution run folder — _generate_final_report reads research + cloud
# outputs from the run folder and writes final-report.md inside it.
# ═══════════════════════════════════════════════════════════════════════════


class RunDirFinalReportTests(unittest.TestCase):
    """With ``run_dir`` set, ``_generate_final_report`` reads research + cloud
    outputs from the run folder and writes ``final-report.md`` inside it."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.temp = Path(self._tmp.name)
        (self.temp / "outputs").mkdir(parents=True)
        self.run_dir = self.temp / "outputs" / "run-x-20260718-120000"
        self.run_dir.mkdir(parents=True)

        import dynamic_cloud.executor as _exc
        _exc._HAS_REPORT_LLM = True
        if not hasattr(_exc, "_report_llm_client"):
            _exc._report_llm_client = mock.MagicMock()
        if not hasattr(_exc, "_report_llm_model"):
            _exc._report_llm_model = lambda: "test-model"

    def tearDown(self) -> None:
        import dynamic_cloud.executor as _exc
        _exc._HAS_REPORT_LLM = False
        for _name in ("_report_llm_client", "_report_llm_model"):
            if hasattr(_exc, _name):
                delattr(_exc, _name)
        self._tmp.cleanup()

    def _populate_run(self) -> None:
        (self.run_dir / "report.md").write_text(
            "# Research\n\nSome findings.\n\n## Sources\n\n- https://example.com/a\n",
            encoding="utf-8",
        )
        out = self.run_dir / "outputs"
        out.mkdir(parents=True)
        (out / "metrics.json").write_text(
            json.dumps({"final_train_accuracy": 0.9}), encoding="utf-8"
        )
        (out / "environment.json").write_text(
            json.dumps({"python_version": "3.11"}), encoding="utf-8"
        )

    def test_final_report_written_inside_run_dir(self) -> None:
        from dynamic_cloud.executor import _generate_final_report
        import dynamic_cloud.executor as _exc

        self._populate_run()
        report_content = "# Unified Report\n\nSynthesis."
        fake_client = mock.MagicMock()
        fake_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content=json.dumps({"report_markdown": report_content})
            ))]
        )

        with mock.patch.object(_exc, "_report_llm_client", return_value=fake_client, create=True), \
             mock.patch.object(_exc, "_report_llm_model", return_value="test-model"):
            result = _generate_final_report("anyjob", run_dir=self.run_dir)

        self.assertEqual(result, "ok")
        # final-report.md is inside the run folder, not at the global path.
        self.assertTrue((self.run_dir / "final-report.md").exists())
        self.assertFalse((self.temp / "outputs" / "final-report.md").exists())
        written = (self.run_dir / "final-report.md").read_text(encoding="utf-8")
        self.assertIn(report_content, written)

    def test_final_report_reads_research_from_run_dir(self) -> None:
        from dynamic_cloud.executor import _generate_final_report
        import dynamic_cloud.executor as _exc

        self._populate_run()
        captured_prompt: dict[str, str] = {}

        def fake_create(**kwargs):
            captured_prompt["content"] = kwargs["messages"][1]["content"]
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                content=json.dumps({"report_markdown": "# x"})
            ))])

        fake_client = mock.MagicMock()
        fake_client.chat.completions.create.side_effect = fake_create

        with mock.patch.object(_exc, "_report_llm_client", return_value=fake_client, create=True), \
             mock.patch.object(_exc, "_report_llm_model", return_value="test-model"):
            _generate_final_report("anyjob", run_dir=self.run_dir)

        # The research report from the run folder was fed to the LLM.
        self.assertIn("Some findings", captured_prompt["content"])

    def test_plot_links_relative_to_run_dir(self) -> None:
        from dynamic_cloud.executor import _generate_final_report
        import dynamic_cloud.executor as _exc

        self._populate_run()
        plots = self.run_dir / "outputs" / "plots"
        plots.mkdir(parents=True)
        (plots / "loss.png").write_text("png", encoding="utf-8")

        # LLM embeds nothing → the plot is force-appended with a relative link.
        fake_client = mock.MagicMock()
        fake_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content=json.dumps({"report_markdown": "# Report\n\nNo plots inline."})
            ))]
        )

        with mock.patch.object(_exc, "_report_llm_client", return_value=fake_client, create=True), \
             mock.patch.object(_exc, "_report_llm_model", return_value="test-model"):
            _generate_final_report("anyjob", run_dir=self.run_dir)

        written = (self.run_dir / "final-report.md").read_text(encoding="utf-8")
        # Link is relative to the run folder (where final-report.md lives).
        self.assertIn("](outputs/plots/loss.png)", written)


if __name__ == "__main__":
    unittest.main()