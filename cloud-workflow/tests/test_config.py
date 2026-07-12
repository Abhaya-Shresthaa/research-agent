from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dynamic_cloud.config import load_amd_settings


class AmdSettingsTests(unittest.TestCase):
    def test_default_vm_size_populates_vm_sizes(self) -> None:
        env = {
            "AMD_API_KEY": "token",
            "AMD_DEFAULT_VM_SIZE": "gpu-mi300x8-1536gb-devcloud",
        }
        with mock.patch.dict(os.environ, env, clear=True), mock.patch("dynamic_cloud.config.load_environment"), mock.patch(
            "dynamic_cloud.config._validate_ssh_private_key"
        ):
            settings = load_amd_settings()

        self.assertEqual(settings.default_size, "gpu-mi300x8-1536gb-devcloud")
        self.assertEqual(settings.vm_sizes, ("gpu-mi300x8-1536gb-devcloud",))

    def test_vm_sizes_env_can_define_fallback_order(self) -> None:
        env = {
            "AMD_API_KEY": "token",
            "AMD_DEFAULT_VM_SIZE": "gpu-mi300x1-192gb-devcloud",
            "AMD_VM_SIZES": " gpu-mi300x8-1536gb-devcloud, gpu-mi300x1-192gb-devcloud ",
        }
        with mock.patch.dict(os.environ, env, clear=True), mock.patch("dynamic_cloud.config.load_environment"), mock.patch(
            "dynamic_cloud.config._validate_ssh_private_key"
        ):
            settings = load_amd_settings()

        self.assertEqual(settings.vm_sizes, ("gpu-mi300x8-1536gb-devcloud", "gpu-mi300x1-192gb-devcloud"))

    def test_region_fallback_order_starts_with_configured_region(self) -> None:
        env = {
            "AMD_API_KEY": "token",
            "AMD_REGION": "atl1",
        }
        with mock.patch.dict(os.environ, env, clear=True), mock.patch("dynamic_cloud.config.load_environment"), mock.patch(
            "dynamic_cloud.config._validate_ssh_private_key"
        ):
            settings = load_amd_settings()

        self.assertEqual(settings.region, "atl1")
        self.assertEqual(settings.regions, ("atl1",))

    def test_regions_env_can_define_fallback_order(self) -> None:
        env = {
            "AMD_API_KEY": "token",
            "AMD_REGION": "atl1",
            "AMD_REGIONS": " atl1, atl1 ",
        }
        with mock.patch.dict(os.environ, env, clear=True), mock.patch("dynamic_cloud.config.load_environment"), mock.patch(
            "dynamic_cloud.config._validate_ssh_private_key"
        ):
            settings = load_amd_settings()

        self.assertEqual(settings.regions, ("atl1",))

    def test_digitalocean_access_token_alias_is_accepted(self) -> None:
        with mock.patch.dict(os.environ, {"DIGITALOCEAN_ACCESS_TOKEN": "token"}, clear=True), mock.patch("dynamic_cloud.config.load_environment"), mock.patch(
            "dynamic_cloud.config._validate_ssh_private_key"
        ):
            settings = load_amd_settings()

        self.assertEqual(settings.api_key, "token")

    def test_amd_token_alias_is_accepted(self) -> None:
        with mock.patch.dict(os.environ, {"AMD_TOKEN": "token"}, clear=True), mock.patch("dynamic_cloud.config.load_environment"), mock.patch(
            "dynamic_cloud.config._validate_ssh_private_key"
        ):
            settings = load_amd_settings()

        self.assertEqual(settings.api_key, "token")

    def test_missing_api_key_mentions_token_alias(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch("dynamic_cloud.config.load_environment"):
            with self.assertRaisesRegex(RuntimeError, "AMD_TOKEN"):
                load_amd_settings()

    def test_inline_public_key_is_loaded(self) -> None:
        env = {
            "AMD_TOKEN": "token",
            "AMD_SSH_PUBLIC_KEY": "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQCfake test@example",
        }
        with mock.patch.dict(os.environ, env, clear=True), mock.patch("dynamic_cloud.config.load_environment"), mock.patch(
            "dynamic_cloud.config._validate_ssh_private_key"
        ):
            settings = load_amd_settings()

        self.assertEqual(settings.ssh_public_key, env["AMD_SSH_PUBLIC_KEY"])

    def test_empty_default_size_raises(self) -> None:
        with mock.patch.dict(os.environ, {"AMD_API_KEY": "token", "AMD_DEFAULT_VM_SIZE": "   "}, clear=True), mock.patch("dynamic_cloud.config.load_environment"):
            with self.assertRaisesRegex(RuntimeError, "AMD_DEFAULT_VM_SIZE"):
                load_amd_settings()

    def test_open_private_key_permissions_raise(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            key_path = Path(temp_dir) / "id_ed25519"
            key_path.write_text("fake-key", encoding="utf-8")
            key_path.chmod(0o644)
            env = {
                "AMD_API_KEY": "token",
                "AMD_SSH_PRIVATE_KEY_PATH": str(key_path),
            }
            with mock.patch.dict(os.environ, env, clear=True), mock.patch("dynamic_cloud.config.load_environment"):
                with self.assertRaisesRegex(PermissionError, "chmod 600"):
                    load_amd_settings()


if __name__ == "__main__":
    unittest.main()
