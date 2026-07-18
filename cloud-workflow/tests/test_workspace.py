from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path

from dynamic_cloud.workspace import create_run_dir, prepare_workspace


class PrepareWorkspaceArchiveTests(unittest.TestCase):
    """E1: ``prepare_workspace(reset=True)`` archives a prior root instead of
    wiping it, so a re-run never silently destroys the prior prepared workspace.

    The workspace root is ``run_dir / "runtime_generated"``; the archive is a
    timestamped sibling of that root."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.run_dir = self.tmp / "outputs" / "myrun"
        self.run_dir.mkdir(parents=True)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    @staticmethod
    def _spec(job_id: str = "train-mnist") -> dict:
        return {"job_id": job_id, "dataset": {"type": "none"}}

    def _archives(self) -> list[Path]:
        rg = self.run_dir / "runtime_generated"
        if not rg.parent.exists():
            return []
        return [
            p for p in self.run_dir.iterdir()
            if p.name.startswith("runtime_generated.") and p.name.endswith(".archive")
        ]

    def test_reset_archives_prior_root(self) -> None:
        ws = prepare_workspace(self._spec(), run_dir=self.run_dir, reset=True)
        # Simulate a prior run's artifact inside the workspace root.
        (ws.root / "payload" / "generated" / "old_script.py").write_text(
            "print('prior')", encoding="utf-8"
        )

        # Re-run with the same run_dir and reset=True.
        prepare_workspace(self._spec(), run_dir=self.run_dir, reset=True)

        # The prior root was archived, not deleted.
        archives = self._archives()
        self.assertEqual(len(archives), 1)
        self.assertTrue(
            (archives[0] / "payload" / "generated" / "old_script.py").exists(),
            "prior workspace artifact should survive in the archive",
        )

    def test_fresh_run_creates_no_archive(self) -> None:
        prepare_workspace(self._spec("fresh-job"), run_dir=self.run_dir, reset=True)
        self.assertEqual(self._archives(), [])

    def test_reset_false_preserves_prior_content(self) -> None:
        ws = prepare_workspace(self._spec(), run_dir=self.run_dir, reset=True)
        (ws.root / "marker.txt").write_text("keep me", encoding="utf-8")

        ws2 = prepare_workspace(self._spec(), run_dir=self.run_dir, reset=False)
        self.assertTrue((ws2.root / "marker.txt").exists())
        self.assertEqual(self._archives(), [])


class CreateRunDirTests(unittest.TestCase):
    """Per-execution run folder naming: ``<slug>-<MM-DD-HHMM>``, unique on
    collision, with a ``run`` fallback for blank labels. The slug is the
    LLM ``job_title`` (kept as-is when short), condensed for long labels."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.outputs_root = Path(self._tmp.name) / "outputs"
        self.outputs_root.mkdir(parents=True)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_name_is_slug_plus_timestamp(self) -> None:
        run_dir = create_run_dir(self.outputs_root, "Train MNIST CNN!")
        self.assertTrue(run_dir.is_dir())
        self.assertEqual(run_dir.parent, self.outputs_root)
        # slug is filesystem-safe (spaces → hyphens, casing preserved) and a
        # -<MM-DD-HHMM> suffix is present.
        name = run_dir.name
        self.assertTrue(
            re.match(r"^Train-MNIST-CNN-\d{2}-\d{2}-\d{4}$", name),
            f"name should be slug + MM-DD-HHMM, got {name!r}",
        )

    def test_blank_label_falls_back_to_run(self) -> None:
        run_dir = create_run_dir(self.outputs_root, "   ")
        self.assertTrue(run_dir.name.startswith("run-"))
        self.assertTrue(run_dir.is_dir())

    def test_collision_gets_numeric_suffix(self) -> None:
        first = create_run_dir(self.outputs_root, "same query")
        # Create another run_dir with the same label and confirm it differs
        # and is a sibling (a -2 suffix is appended on a same-minute collision).
        second = create_run_dir(self.outputs_root, "same query")
        self.assertNotEqual(first.name, second.name)
        self.assertEqual(first.parent, second.parent)
        self.assertTrue(second.is_dir())
        self.assertTrue(first.is_dir())  # first not clobbered

    def test_long_label_is_truncated(self) -> None:
        long_label = "a" * 200
        run_dir = create_run_dir(self.outputs_root, long_label)
        # Strip the trailing -<MM-DD-HHMM>[-<n>] timestamp; the slug that
        # remains must be capped at ~30 chars.
        slug_part = re.sub(r"-\d{2}-\d{2}-\d{4}(-\d+)?$", "", run_dir.name)
        self.assertLessEqual(len(slug_part), 30)
        self.assertTrue(slug_part.startswith("a"))


if __name__ == "__main__":
    unittest.main()
