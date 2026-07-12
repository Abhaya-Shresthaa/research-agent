from __future__ import annotations

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

    def _workspace(self) -> JobWorkspace:
        root = Path(tempfile.mkdtemp())
        payload = root / "payload"
        generated = payload / "generated"
        data = payload / "data"
        generated.mkdir(parents=True)
        data.mkdir()
        (payload / "metadata.json").write_text("{}", encoding="utf-8")
        return JobWorkspace("job", root, payload, generated, data)

    def _settings(self) -> AmdSettings:
        return AmdSettings(api_key="token", ssh_private_key_path=Path("/tmp/key"))


if __name__ == "__main__":
    unittest.main()
