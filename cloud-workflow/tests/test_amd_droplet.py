from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock
import subprocess

import requests

from dynamic_cloud.amd_droplet import AmdDropletManager
from dynamic_cloud.config import AmdSettings


class FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text
        self.headers = {}

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(response=self)


class AmdDropletProjectTests(unittest.TestCase):
    def test_no_project_id_skips_assignment(self) -> None:
        manager = AmdDropletManager(AmdSettings(api_key="token", ssh_private_key_path=Path("/tmp/key")))
        with mock.patch.object(manager, "ensure_ssh_key", return_value=99) as ensure, mock.patch(
            "dynamic_cloud.amd_droplet.requests.post",
            return_value=FakeResponse(202, {"droplet": {"id": 123}}),
        ) as post:
            manager.create_droplet()

        self.assertEqual(post.call_count, 1)
        ensure.assert_not_called()
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["ssh_keys"], [])
        self.assertEqual(payload["tags"], [])
        self.assertFalse(payload["backups"])
        self.assertFalse(payload["ipv6"])
        self.assertFalse(payload["monitoring"])
        self.assertIn("#cloud-config", payload["user_data"])
        self.assertIn("docker build --pull=false", payload["user_data"])
        self.assertNotIn("docker pull", payload["user_data"])
        self.assertEqual(payload["vpc_uuid"], "8244dc95-5e6a-45b1-a667-d8c80a851d9b")
        self.assertEqual(post.call_args.args[0], f"{manager.base_url}/droplets")

    def test_project_id_assigns_created_droplet(self) -> None:
        settings = AmdSettings(api_key="token", project_id="proj-1", ssh_private_key_path=Path("/tmp/key"))
        manager = AmdDropletManager(settings)
        responses = [
            FakeResponse(202, {"droplet": {"id": 123}}),
            FakeResponse(200, {"resources": []}),
        ]
        with mock.patch.object(manager, "ensure_ssh_key", return_value=99), mock.patch(
            "dynamic_cloud.amd_droplet.requests.post",
            side_effect=responses,
        ) as post:
            manager.create_droplet()

        self.assertEqual(post.call_args_list[1].args[0], f"{manager.BASE_URL}/projects/proj-1/resources")
        self.assertEqual(post.call_args_list[1].kwargs["json"], {"resources": ["do:droplet:123"]})

    def test_project_assignment_failure_warns_and_continues(self) -> None:
        settings = AmdSettings(api_key="token", project_id="proj-1", ssh_private_key_path=Path("/tmp/key"))
        manager = AmdDropletManager(settings)
        responses = [
            FakeResponse(202, {"droplet": {"id": 123}}),
            FakeResponse(500, text="boom"),
        ]
        with mock.patch.object(manager, "ensure_ssh_key", return_value=99), mock.patch(
            "dynamic_cloud.amd_droplet.requests.post",
            side_effect=responses,
        ), mock.patch("builtins.print") as print_mock:
            manager.create_droplet()

        self.assertEqual(manager.droplet_id, 123)
        self.assertTrue(any("Warning: could not assign Droplet" in str(call) for call in print_mock.call_args_list))

    def test_create_droplet_falls_back_to_next_region_when_size_unavailable(self) -> None:
        settings = AmdSettings(
            api_key="token",
            region="atl1",
            regions=("atl1", "atl2"),
            ssh_private_key_path=Path("/tmp/key"),
        )
        manager = AmdDropletManager(settings)
        responses = [
            FakeResponse(422, {"message": "Size is not available in this region."}),
            FakeResponse(202, {"droplet": {"id": 456}}),
        ]

        with mock.patch.object(manager, "ensure_ssh_key", return_value=99), mock.patch(
            "dynamic_cloud.amd_droplet.requests.post",
            side_effect=responses,
        ) as post:
            manager.create_droplet()

        self.assertEqual(manager.droplet_id, 456)
        self.assertEqual(manager.region, "atl2")
        self.assertEqual(post.call_args_list[0].kwargs["json"]["region"], "atl1")
        self.assertEqual(post.call_args_list[1].kwargs["json"]["region"], "atl2")

    def test_explicit_ssh_keys_are_used_without_registration(self) -> None:
        settings = AmdSettings(
            api_key="token",
            ssh_keys=(99,),
            tags=("dynamic-ml",),
            vpc_uuid="8244dc95-5e6a-45b1-a667-d8c80a851d9b",
            ssh_private_key_path=Path("/tmp/key"),
        )
        manager = AmdDropletManager(settings)

        with mock.patch.object(manager, "ensure_ssh_key", return_value=100) as ensure, mock.patch(
            "dynamic_cloud.amd_droplet.requests.post",
            return_value=FakeResponse(202, {"droplet": {"id": 123}}),
        ) as post:
            manager.create_droplet()

        ensure.assert_not_called()
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["ssh_keys"], [99])
        self.assertEqual(payload["tags"], ["dynamic-ml"])
        self.assertEqual(payload["vpc_uuid"], "8244dc95-5e6a-45b1-a667-d8c80a851d9b")

    def test_inline_public_key_registers_when_named_key_is_missing(self) -> None:
        settings = AmdSettings(
            api_key="token",
            ssh_key_name="Dynamic_Workflow",
            ssh_public_key="ssh-rsa dGVzdA== test@example",
            ssh_private_key_path=Path("/tmp/key"),
        )
        manager = AmdDropletManager(settings)
        responses = [
            FakeResponse(200, {"ssh_keys": []}),
            FakeResponse(200, {"ssh_keys": []}),
            FakeResponse(201, {"ssh_key": {"id": 99, "fingerprint": "09:8f:6b:cd:46:21:d3:73:ca:de:4e:83:26:27:b4:f6"}}),
        ]

        with mock.patch("dynamic_cloud.amd_droplet.requests.get", side_effect=responses[:2]) as get, mock.patch(
            "dynamic_cloud.amd_droplet.requests.post",
            return_value=responses[2],
        ) as post:
            key_id = manager.ensure_ssh_key()

        self.assertEqual(key_id, 99)
        self.assertEqual(get.call_count, 2)
        self.assertEqual(post.call_args.kwargs["json"]["name"], "Dynamic_Workflow")
        self.assertEqual(post.call_args.kwargs["json"]["public_key"], settings.ssh_public_key)

    def test_key_lookup_falls_back_to_amd_digitalocean_route(self) -> None:
        settings = AmdSettings(api_key="token", ssh_private_key_path=Path("/tmp/key"))
        manager = AmdDropletManager(settings)
        responses = [
            FakeResponse(404, text="not found"),
            FakeResponse(200, {"ssh_keys": [{"id": 123, "name": "Dynamic_Workflow"}]}),
        ]

        with mock.patch("dynamic_cloud.amd_droplet.requests.get", side_effect=responses) as get:
            key_id = manager._find_ssh_key("Dynamic_Workflow")

        self.assertEqual(key_id, 123)
        self.assertEqual(manager.base_url, manager.FALLBACK_BASE_URL)
        self.assertEqual(get.call_args_list[1].args[0], f"{manager.FALLBACK_BASE_URL}/account/keys")

    def test_wait_for_ssh_requires_successful_login(self) -> None:
        settings = AmdSettings(
            api_key="token",
            ssh_private_key_path=Path("/tmp/key"),
            wait_ssh_timeout_sec=30,
        )
        manager = AmdDropletManager(settings)
        attempts = [
            subprocess.CompletedProcess(args=["ssh"], returncode=255, stdout="", stderr="Connection refused"),
            subprocess.CompletedProcess(args=["ssh"], returncode=0, stdout="", stderr=""),
        ]

        with mock.patch.object(manager, "_is_tcp_ssh_open", return_value=True), mock.patch(
            "dynamic_cloud.amd_droplet.subprocess.run",
            side_effect=attempts,
        ) as run_mock, mock.patch("dynamic_cloud.amd_droplet.time.sleep"):
            manager.wait_for_ssh("10.0.0.4")

        self.assertEqual(run_mock.call_count, 2)


if __name__ == "__main__":
    unittest.main()
