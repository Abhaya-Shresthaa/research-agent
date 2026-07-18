from __future__ import annotations

import tempfile
import unittest
import subprocess
from pathlib import Path
from unittest import mock
from types import SimpleNamespace

from dynamic_cloud.docker_runner import RemoteHostRunner


class RemoteHostRunnerTests(unittest.TestCase):
    def test_redact_command_hides_ssh_key(self) -> None:
        redacted = RemoteHostRunner._redact_command(
            ["ssh", "-i", "/secret/id_rsa", "root@10.0.0.4"]
        )

        self.assertEqual(redacted, ["ssh", "-i", "<SSH_KEY>", "root@10.0.0.4"])

    def test_redact_command_no_key_leaves_unchanged(self) -> None:
        cmd = ["ssh", "root@10.0.0.4", "true"]

        redacted = RemoteHostRunner._redact_command(cmd)

        self.assertEqual(redacted, cmd)

    def test_max_ssh_attempts_defaults_to_5(self) -> None:
        attempts = RemoteHostRunner._max_ssh_attempts(
            ["ssh", "root@10.0.0.4", "true"], capture=True
        )

        self.assertEqual(attempts, 5)

    def test_max_ssh_attempts_is_1_for_non_ssh_commands(self) -> None:
        attempts = RemoteHostRunner._max_ssh_attempts(["true"], capture=True)

        self.assertEqual(attempts, 1)

    def test_transient_ssh_error_detects_connection_refused(self) -> None:
        result = mock.Mock(returncode=255, stderr="Connection refused")

        self.assertTrue(RemoteHostRunner._is_transient_ssh_error(result))

    def test_transient_ssh_error_accepts_subprocess_bytes_stderr(self) -> None:
        result = mock.Mock(returncode=255, stderr=b"ssh: connect to host example port 22: Connection refused")

        self.assertTrue(RemoteHostRunner._is_transient_ssh_error(result))

    def test_transient_ssh_error_non_transient(self) -> None:
        result = mock.Mock(returncode=1, stderr="Permission denied")

        self.assertFalse(RemoteHostRunner._is_transient_ssh_error(result))

    def test_remote_root_root_user(self) -> None:
        settings = SimpleNamespace(admin_user="root", ssh_private_key_path="/tmp/key")
        runner = RemoteHostRunner(settings)

        self.assertEqual(runner.remote_root, "/root/dynamic_jobs")

    def test_remote_root_non_root_user(self) -> None:
        settings = SimpleNamespace(admin_user="ubuntu", ssh_private_key_path="/tmp/key")
        runner = RemoteHostRunner(settings)

        self.assertEqual(runner.remote_root, "/home/ubuntu/dynamic_jobs")

    def test_remote_root_can_be_overridden(self) -> None:
        settings = SimpleNamespace(admin_user="root", ssh_private_key_path="/tmp/key")
        runner = RemoteHostRunner(settings, remote_root="/custom/path")

        self.assertEqual(runner.remote_root, "/custom/path")

    def test_container_name_is_docker_safe_bounded_and_distinct(self) -> None:
        first = RemoteHostRunner._container_name("/root/dynamic_jobs/Beans CNN!", "dataset-inspect")
        second = RemoteHostRunner._container_name("/root/dynamic_jobs/beans-cnn", "dataset-inspect")

        self.assertRegex(first, r"^[a-z0-9][a-z0-9_.-]*$")
        self.assertLessEqual(len(first), 63)
        self.assertNotEqual(first, second)
        self.assertLessEqual(len(RemoteHostRunner._container_name("/job", "x" * 200)), 63)

    def test_dataset_cache_is_shared_outside_job_workspaces(self) -> None:
        settings = SimpleNamespace(admin_user="root", ssh_private_key_path="/tmp/key")
        runner = RemoteHostRunner(settings, remote_root="/custom/dynamic_jobs")
        commands: list[str] = []

        def record(_ip: str, command: str, **_kwargs):
            commands.append(command)
            return subprocess.CompletedProcess(["ssh"], 0, "", "")

        with mock.patch.object(runner, "_ssh", side_effect=record):
            runner._restore_dataset_cache("203.0.113.10", "dynamic-cloud-job-runtime")
            runner._save_dataset_cache("203.0.113.10", "dynamic-cloud-job-runtime")

        self.assertEqual(runner._remote_dataset_cache_dir(), "/custom/dynamic_jobs/datasets")
        self.assertIn("/custom/dynamic_jobs/datasets", commands[0])
        self.assertIn("/workspace/prepared_datasets", commands[0])
        self.assertIn("/custom/dynamic_jobs/datasets", commands[1])
        self.assertIn("/workspace/prepared_datasets", commands[1])

    def test_container_stage_uses_shared_helpers_for_inspection_and_training(self) -> None:
        settings = SimpleNamespace(admin_user="root", ssh_private_key_path="/tmp/key")
        runner = RemoteHostRunner(settings)

        with mock.patch.object(runner, "_wait_for_stable_ssh"), mock.patch.object(
            runner, "_run_container_stage"
        ) as stage:
            runner.run_dataset_inspection("203.0.113.10", "/root/dynamic_jobs/job")
            runner.execute_bootstrap("203.0.113.10", "/root/dynamic_jobs/job")

        self.assertEqual(stage.call_count, 2)
        self.assertEqual(stage.call_args_list[0].args[2], "dataset-inspect")
        self.assertTrue(stage.call_args_list[0].kwargs["install_packages"])
        self.assertEqual(stage.call_args_list[1].args[2], "runtime")
        self.assertFalse(stage.call_args_list[1].kwargs["install_packages"])

    def test_metadata_marker_extraction_ignores_login_banner(self) -> None:
        text = "Please wait while we get your droplet ready...\n__DYNAMIC_CLOUD_JSON_BEGIN__\neyJvayI6IHRydWV9\n__DYNAMIC_CLOUD_JSON_END__\n"

        self.assertEqual(
            RemoteHostRunner._between_markers(text, "__DYNAMIC_CLOUD_JSON_BEGIN__", "__DYNAMIC_CLOUD_JSON_END__"),
            "eyJvayI6IHRydWV9",
        )

    def test_initialization_retries_transient_ssh_restart_until_cloud_init_completes(self) -> None:
        settings = SimpleNamespace(
            admin_user="root",
            ssh_private_key_path="/tmp/key",
            wait_initialization_timeout_sec=120,
        )
        runner = RemoteHostRunner(settings)
        attempts = [
            subprocess.CompletedProcess(["ssh"], 255, "", "Connection reset by peer"),
            subprocess.CompletedProcess(["ssh"], 0, "status: done\n", ""),
        ]

        with mock.patch("dynamic_cloud.docker_runner.subprocess.run", side_effect=attempts) as run, mock.patch(
            "dynamic_cloud.docker_runner.time.sleep"
        ):
            runner._wait_for_vm_initialization("203.0.113.10")

        self.assertEqual(run.call_count, 2)

    def test_initialization_surfaces_cloud_init_failure_without_retrying(self) -> None:
        settings = SimpleNamespace(
            admin_user="root",
            ssh_private_key_path="/tmp/key",
            wait_initialization_timeout_sec=120,
        )
        runner = RemoteHostRunner(settings)
        failed = subprocess.CompletedProcess(["ssh"], 1, "status: error\n", "")

        with mock.patch("dynamic_cloud.docker_runner.subprocess.run", return_value=failed) as run:
            with self.assertRaisesRegex(RuntimeError, "VM initialization failed"):
                runner._wait_for_vm_initialization("203.0.113.10")

        self.assertEqual(run.call_count, 1)

    def test_stable_ssh_waits_past_final_reboot_then_requires_three_probes(self) -> None:
        settings = SimpleNamespace(
            admin_user="root",
            ssh_private_key_path="/tmp/key",
            wait_initialization_timeout_sec=120,
        )
        runner = RemoteHostRunner(settings)
        attempts = [
            subprocess.CompletedProcess(["ssh"], 255, "", "Connection refused"),
            subprocess.CompletedProcess(["ssh"], 0, "", ""),
            subprocess.CompletedProcess(["ssh"], 0, "", ""),
            subprocess.CompletedProcess(["ssh"], 0, "", ""),
        ]

        with mock.patch("dynamic_cloud.docker_runner.subprocess.run", side_effect=attempts) as run, mock.patch(
            "dynamic_cloud.docker_runner.time.sleep"
        ):
            runner._wait_for_stable_ssh("203.0.113.10")

        self.assertEqual(run.call_count, 4)

    # ── D2: configurable SSH / provisioning timeouts ──────────────────────

    def test_wait_for_ssh_ready_attempts_derived_from_settings(self) -> None:
        # wait_ssh_timeout_sec=30 → clamped to 60s → 60//10 = 6 attempts.
        settings = SimpleNamespace(
            admin_user="root",
            ssh_private_key_path="/tmp/key",
            wait_ssh_timeout_sec=30,
        )
        runner = RemoteHostRunner(settings)
        refused = subprocess.CompletedProcess(["ssh"], 255, "", "Connection refused")

        with mock.patch("dynamic_cloud.docker_runner.subprocess.run", return_value=refused) as run, mock.patch(
            "dynamic_cloud.docker_runner.time.sleep"
        ):
            with self.assertRaises(TimeoutError):
                runner._wait_for_ssh_ready("203.0.113.10")

        self.assertEqual(run.call_count, 6)

    def test_wait_for_provisioning_attempts_derived_from_settings(self) -> None:
        # wait_provisioning_timeout_sec=120 → 120//10 = 12 attempts.
        settings = SimpleNamespace(
            admin_user="root",
            ssh_private_key_path="/tmp/key",
            wait_provisioning_timeout_sec=120,
        )
        runner = RemoteHostRunner(settings)
        # Banner still present → never ready.
        still_waiting = subprocess.CompletedProcess(["ssh"], 0, "Please wait while we get your droplet ready\n", "")

        with mock.patch("dynamic_cloud.docker_runner.subprocess.run", return_value=still_waiting) as run, mock.patch(
            "dynamic_cloud.docker_runner.time.sleep"
        ):
            with self.assertRaises(TimeoutError):
                runner._wait_for_provisioning("203.0.113.10")

        self.assertEqual(run.call_count, 12)

    def test_wait_for_provisioning_succeeds_when_banner_gone(self) -> None:
        settings = SimpleNamespace(
            admin_user="root",
            ssh_private_key_path="/tmp/key",
            wait_provisioning_timeout_sec=600,
        )
        runner = RemoteHostRunner(settings)
        ready = subprocess.CompletedProcess(["ssh"], 0, "SSHD_READY\n", "")

        with mock.patch("dynamic_cloud.docker_runner.subprocess.run", return_value=ready) as run, mock.patch(
            "dynamic_cloud.docker_runner.time.sleep"
        ):
            runner._wait_for_provisioning("203.0.113.10")

        self.assertEqual(run.call_count, 1)

    # ── D3: probe 0 — AMD Quick Start conda path ──────────────────────────

    def test_discover_remote_python_quick_conda_path(self) -> None:
        settings = SimpleNamespace(admin_user="root", ssh_private_key_path="/tmp/key")
        runner = RemoteHostRunner(settings)

        def fake_ssh(_ip: str, command: str, **_kwargs):
            # Probe 0 first path succeeds.
            if "/opt/conda/envs/pytorch/bin/python3" in command:
                return subprocess.CompletedProcess(["ssh"], 0, "1\n", "")
            return subprocess.CompletedProcess(["ssh"], 1, "", "")

        with mock.patch.object(runner, "_ssh", side_effect=fake_ssh):
            py = runner._discover_remote_python("203.0.113.10")

        self.assertEqual(py, "/opt/conda/envs/pytorch/bin/python3")
        self.assertEqual(runner._remote_python, "/opt/conda/envs/pytorch/bin/python3")

    def test_discover_remote_python_falls_through_when_quick_path_missing(self) -> None:
        settings = SimpleNamespace(admin_user="root", ssh_private_key_path="/tmp/key")
        runner = RemoteHostRunner(settings)

        def fake_ssh(_ip: str, command: str, **_kwargs):
            # Probe 0 paths fail, probe 1 (system python3) succeeds.
            if "python3 -c 'import torch" in command and "/opt/conda" not in command:
                return subprocess.CompletedProcess(["ssh"], 0, "1\n", "")
            return subprocess.CompletedProcess(["ssh"], 1, "", "")

        with mock.patch.object(runner, "_ssh", side_effect=fake_ssh):
            py = runner._discover_remote_python("203.0.113.10")

        self.assertEqual(py, "python3")

    # ── D4: download_outputs packs on the VM, scp's the tarball, then extracts ──

    def test_download_outputs_packs_scps_and_extracts(self) -> None:
        settings = SimpleNamespace(admin_user="root", ssh_private_key_path="/tmp/key")
        runner = RemoteHostRunner(settings)
        captured: dict = {"ssh_cmds": [], "scp": None, "extract": None}

        def fake_ssh(_ip, command, **_kwargs):
            captured["ssh_cmds"].append(command)
            # Pack command reports a 1234-byte tarball size via `stat`.
            if command.startswith("cd ") and "stat -c %s" in command:
                return subprocess.CompletedProcess(["ssh"], 0, "1234\n", "")
            return subprocess.CompletedProcess(["ssh"], 0, "", "")

        def fake_run(cmd, **kwargs):
            # The scp invocation — copy the "remote" tarball to the local path.
            if cmd and cmd[0] == "scp":
                captured["scp"] = cmd
                local_path = Path(cmd[-1])
                local_path.parent.mkdir(parents=True, exist_ok=True)
                local_path.write_bytes(b"x" * 1234)
                return subprocess.CompletedProcess(cmd, 0, "", "")
            return subprocess.CompletedProcess(cmd, 0, "", "")

        def fake_subprocess_run(cmd, **kwargs):
            captured["extract"] = cmd
            # Simulate tar extracting `outputs/run_manifest.json` into -C target.
            extract_dir = Path(cmd[cmd.index("-C") + 1])
            (extract_dir / "outputs").mkdir(parents=True, exist_ok=True)
            (extract_dir / "outputs" / "run_manifest.json").write_text("{}")
            return subprocess.CompletedProcess(cmd, 0, "", "")

        _run_dir = Path(tempfile.mkdtemp())
        workspace = SimpleNamespace(
            job_id="job-x",
            payload_dir=Path("/tmp/payload"),
            outputs_dir=_run_dir / "outputs",
        )

        with mock.patch.object(runner, "_ssh", side_effect=fake_ssh), \
             mock.patch.object(runner, "_run", side_effect=fake_run), \
             mock.patch("dynamic_cloud.docker_runner.subprocess.run",
                        side_effect=fake_subprocess_run):
            result = runner.download_outputs("203.0.113.10", workspace, "/root/dynamic_jobs/job")

        # 1. Remote pack: tar to a *file* (not stdout `-`), stderr to a file,
        #    and a `stat -c %s` to report the tarball size.
        pack_cmd = captured["ssh_cmds"][0]
        self.assertIn("tar -czf ", pack_cmd)
        self.assertNotIn("tar -czf -", pack_cmd)
        self.assertIn("outputs.tar.gz", pack_cmd)
        self.assertIn("2>/tmp/dc_tar_pack.err", pack_cmd)
        self.assertIn("stat -c %s", pack_cmd)

        # 2. scp fetches the remote tarball to a local .part file.
        self.assertIsNotNone(captured["scp"])
        self.assertEqual(captured["scp"][0], "scp")
        self.assertIn("outputs.tar.gz", captured["scp"][-2])
        self.assertTrue(captured["scp"][-1].endswith("outputs.tar.gz.part"))

        # 3. Local extraction uses the downloaded file (not stdin `-`).
        self.assertEqual(captured["extract"][0], "tar")
        self.assertEqual(captured["extract"][1], "-xzf")
        self.assertNotEqual(captured["extract"][2], "-")

        # 4. Outputs land in the run folder's outputs/ dir and the path returns.
        self.assertTrue((_run_dir / "outputs" / "run_manifest.json").exists())
        self.assertEqual(result, _run_dir / "outputs")

        # 5. The .part tarball is cleaned up locally after extraction.
        self.assertFalse((_run_dir / "outputs.tar.gz.part").exists())

    # ── D7: banner suppression is non-destructive ─────────────────────────

    def test_silence_shell_banner_backs_up_bashrc_and_avoids_system_files(self) -> None:
        settings = SimpleNamespace(admin_user="root", ssh_private_key_path="/tmp/key")
        runner = RemoteHostRunner(settings)
        captured: dict = {}

        def fake_ssh(_ip, command, **_kwargs):
            captured["cmd"] = command
            return subprocess.CompletedProcess(["ssh"], 0, "", "")

        with mock.patch.object(runner, "_ssh", side_effect=fake_ssh):
            runner._silence_shell_banner("203.0.113.10")

        cmd = captured["cmd"]
        # Back up the original ~/.bashrc once, BEFORE overwriting it.
        self.assertIn("cp ~/.bashrc ~/.bashrc.dc_backup", cmd)
        self.assertIn("[ ! -f ~/.bashrc.dc_backup ]", cmd)
        # New bashrc sources the backup (the user's real config) for interactive shells.
        self.assertIn("~/.bashrc.dc_backup", cmd)
        self.assertIn("return", cmd)  # early return for non-interactive shells
        # Must NOT mutate system files (the old destructive approach).
        self.assertNotIn("sed -i", cmd)
        self.assertNotIn("/etc/profile.d", cmd)
        self.assertNotIn("chmod -x", cmd)
        self.assertNotIn("/etc/update-motd.d", cmd)


if __name__ == "__main__":
    unittest.main()
