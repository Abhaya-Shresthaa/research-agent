from __future__ import annotations

import unittest

from dynamic_cloud.docker_runner import RemoteHostRunner
from dynamic_cloud.runtime_layers import build_runtime_metadata, dataset_package_install_code, package_check_metadata
from dynamic_cloud.workspace import normalize_dataset_config


class RuntimeDependencyCheckTests(unittest.TestCase):
    def test_huggingface_datasets_requires_loader_symbols(self) -> None:
        self.assertEqual(
            package_check_metadata("datasets"),
            {"import": "datasets", "attributes": ["load_dataset", "load_from_disk"]},
        )

    def test_runtime_metadata_includes_package_checks(self) -> None:
        metadata = build_runtime_metadata(
            {"framework": "tensorflow", "dataset": {"type": "huggingface", "id": "beans"}},
            script="print('ok')",
            requested_packages=[],
        )

        self.assertEqual(metadata["extra_packages"], ["datasets"])
        self.assertEqual(
            metadata["package_checks"]["datasets"],
            {"import": "datasets", "attributes": ["load_dataset", "load_from_disk"]},
        )

    def test_dataset_inspection_install_code_checks_datasets_symbols(self) -> None:
        install_code = dataset_package_install_code()

        self.assertIn("'datasets': ['load_dataset', 'load_from_disk']", install_code)
        self.assertIn("hasattr(module, attr)", install_code)
        # --upgrade is intentionally omitted: pre-installed frameworks must not
        # be overwritten on AMD Quick Start Images
        self.assertIn('"pip", "install", *missing', install_code)

    def test_remote_host_runner_remote_root(self) -> None:
        runner = RemoteHostRunner.__new__(RemoteHostRunner)
        runner.remote_root = "/root/dynamic_jobs"

        self.assertEqual(runner.remote_root, "/root/dynamic_jobs")

    def test_unsupported_dataset_type_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_dataset_config({"type": "kaggle", "id": "salader/dogs-vs-cats"})


if __name__ == "__main__":
    unittest.main()
