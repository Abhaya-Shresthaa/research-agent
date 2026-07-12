from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path

from dynamic_cloud.runtime_layers import _runtime_validator_source, build_runtime_metadata


class RuntimeValidatorTests(unittest.TestCase):
    def test_training_job_requires_training_logs(self) -> None:
        metadata = build_runtime_metadata(
            {"framework": "pytorch", "task_type": "training", "dataset": {"type": "none"}},
            "print('ok')",
            [],
        )

        self.assertIn("training_logs.json", metadata["required_outputs"])

    def test_expected_outputs_force_required_artifact(self) -> None:
        metadata = build_runtime_metadata(
            {
                "framework": "pytorch",
                "task_type": "preprocess",
                "dataset": {"type": "none"},
                "expected_outputs": ["predictions.csv", "plots/"],
            },
            "print('ok')",
            [],
        )

        self.assertIn("predictions.csv", metadata["required_outputs"])
        self.assertIn("plots", metadata["required_dirs"])

    def test_evaluation_job_passes_without_plots_or_training_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            outputs = root / "outputs"
            workspace.mkdir()
            outputs.mkdir()
            (workspace / "metadata.json").write_text(
                '{"required_outputs": ["environment.json", "metrics.json"], '
                '"optional_outputs": ["training_logs.json"], "optional_dirs": ["plots"]}',
                encoding="utf-8",
            )
            (outputs / "environment.json").write_text("{}", encoding="utf-8")
            (outputs / "metrics.json").write_text("{}", encoding="utf-8")

            module = types.ModuleType("runtime_validator_under_test")
            exec(_runtime_validator_source(), module.__dict__)
            module.WORKSPACE = workspace
            module.OUTPUTS = outputs

            module.main()

            self.assertTrue((outputs / "logs" / "output_report.txt").exists())


if __name__ == "__main__":
    unittest.main()
