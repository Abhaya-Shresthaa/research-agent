from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import run


class SetupAndInspectLocalDatasetTests(unittest.TestCase):
    """E2: re-staging a local dataset must preserve the previously-staged
    content (moved to ``.uploading_archive/<ts>/``) instead of deleting it,
    and the archive must live outside ``uploading_data/`` so it is excluded
    from the ``local_paths`` listing."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.upload = self.tmp / "uploading_data"
        self.upload.mkdir(parents=True)
        self.archive_root = self.tmp / ".uploading_archive"

        self._patches = [
            mock.patch.object(run, "LOCAL_UPLOAD_DIR", self.upload),
            mock.patch.object(run, "_UPLOADING_ARCHIVE_DIR", self.archive_root),
            mock.patch.object(
                run, "_inspect_local_dataset",
                lambda selected, data_path: {"dataset": selected, "ok": True},
            ),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self) -> None:
        for p in self._patches:
            p.stop()
        self._tmp.cleanup()

    def _new_source(self, name: str, content: str = "new") -> Path:
        src = self.tmp / "sources" / name
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_text(content, encoding="utf-8")
        return src

    def test_prior_staging_content_archived_not_deleted(self) -> None:
        # Prior staged dataset in uploading_data/.
        prior_file = self.upload / "old_data.csv"
        prior_file.write_text("old", encoding="utf-8")
        prior_dir = self.upload / "old_dir"
        prior_dir.mkdir()
        (prior_dir / "inner.txt").write_text("old-dir", encoding="utf-8")

        new_src = self._new_source("new_data.csv", "new")
        selected = {"type": "local", "local_paths": [str(new_src)]}

        run._setup_and_inspect_local_dataset(selected)

        # New content is staged.
        self.assertTrue((self.upload / "new_data.csv").exists())
        self.assertEqual((self.upload / "new_data.csv").read_text(encoding="utf-8"), "new")

        # Old content was moved to the archive, not deleted.
        archives = list(self.archive_root.iterdir()) if self.archive_root.exists() else []
        self.assertEqual(len(archives), 1)
        archived = archives[0]
        self.assertEqual((archived / "old_data.csv").read_text(encoding="utf-8"), "old")
        self.assertTrue((archived / "old_dir" / "inner.txt").exists())

        # The archive is outside uploading_data/, so the contents listing
        # (local_paths) only contains the newly staged file.
        local_paths = selected["local_paths"]
        names = [Path(p).name for p in local_paths]
        self.assertIn("new_data.csv", names)
        self.assertNotIn("old_data.csv", names)
        self.assertFalse(any(".uploading_archive" in p for p in local_paths))

    def test_empty_staging_creates_no_archive(self) -> None:
        new_src = self._new_source("only.csv", "new")
        selected = {"type": "local", "local_paths": [str(new_src)]}

        run._setup_and_inspect_local_dataset(selected)

        self.assertTrue((self.upload / "only.csv").exists())
        self.assertFalse(self.archive_root.exists())

    def test_self_reference_does_not_archive(self) -> None:
        # User points at uploading_data/ itself — existing content is reused,
        # nothing archived, nothing cleared.
        existing = self.upload / "keep.csv"
        existing.write_text("keep", encoding="utf-8")

        selected = {"type": "local", "local_paths": [str(self.upload.resolve())]}
        run._setup_and_inspect_local_dataset(selected)

        self.assertTrue(existing.exists())
        self.assertEqual(existing.read_text(encoding="utf-8"), "keep")
        self.assertFalse(self.archive_root.exists())


if __name__ == "__main__":
    unittest.main()
