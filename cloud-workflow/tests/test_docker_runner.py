from __future__ import annotations

import unittest
import subprocess
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


if __name__ == "__main__":
    unittest.main()
