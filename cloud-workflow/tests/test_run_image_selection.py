from __future__ import annotations

import unittest
from unittest import mock

import run


class AmdDropletImageSelectionTests(unittest.TestCase):
    def test_pytorch_uses_pytorch_quick_start_image(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(
                run._amd_droplet_image_for_framework("pytorch"),
                "amddevelopercloud-pytorch2100rocm724",
            )

    def test_tensorflow_uses_rocm_base_image(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(
                run._amd_droplet_image_for_framework("tensorflow"),
                "rocm-7-2-4",
            )

    def test_framework_specific_env_override_wins(self) -> None:
        with mock.patch.dict("os.environ", {"AMD_PYTORCH_GPU_IMAGE": "custom-pytorch"}, clear=True):
            self.assertEqual(run._amd_droplet_image_for_framework("torch"), "custom-pytorch")


if __name__ == "__main__":
    unittest.main()
