from __future__ import annotations

import unittest

from dynamic_cloud.runtime_layers import _dataset_manager_source, build_runtime_metadata


class RuntimeMetadataTests(unittest.TestCase):
    """D5 — metadata must record the actual native execution mode."""

    def test_metadata_records_native_host_execution_mode(self) -> None:
        md = build_runtime_metadata(
            {"framework": "pytorch", "runtime": {"accelerator": "gpu"}},
            "",
            [],
        )

        self.assertEqual(md["execution_mode"], "native_host")
        self.assertIn("image", md)  # reference slug still present


class DatasetManagerSourceTests(unittest.TestCase):
    """The dataset_manager module is generated as a source string and runs on
    the VM, so we assert on its contents for the D8 / D9 fixes."""

    def setUp(self) -> None:
        self.src = _dataset_manager_source()

    # D8 — git clone must not force `--branch main`; tarball fallback tries
    # the common default branches when the user did not name one.
    def test_github_clone_does_not_force_main_when_branch_unset(self) -> None:
        # The original bug was `branch = config.get("branch") or "main"`
        # followed by an unconditional `--branch $branch`. The fix keeps an
        # unset branch as None so git clones the repo's default branch.
        self.assertIn("requested_branch = config.get(\"branch\")", self.src)
        self.assertNotIn('or "main"', self.src.split("def _prepare_github")[1].split("def _write_info")[0])
        # --branch is only added when a branch was explicitly requested.
        self.assertIn("if requested_branch:", self.src)

    def test_github_tarball_fallback_tries_main_then_master(self) -> None:
        github_body = self.src.split("def _prepare_github")[1].split("def _write_info")[0]
        self.assertIn('"main", "master"', github_body)
        self.assertIn("candidate_branches", github_body)

    def test_github_tarball_fallback_uses_curl_and_strip_components(self) -> None:
        github_body = self.src.split("def _prepare_github")[1].split("def _write_info")[0]
        self.assertIn("curl", github_body)
        self.assertIn("--strip-components=1", github_body)

    # D9 — HuggingFace compat libs are cached in a stable directory under the
    # dataset cache root, not a fresh tempdir on every retry.
    def test_hf_compat_libs_use_stable_cache_dir(self) -> None:
        hf_body = self.src.split("def _retry_hf_dataset_with_compat_libs", 1)[1]
        self.assertIn(".hf_compat_overlay", hf_body)
        self.assertIn("DATASETS_ROOT", hf_body)
        # Reuse check: skip the pip install when the dirs already exist.
        self.assertIn("isdir", hf_body)

    def test_hf_compat_libs_do_not_use_tempfile_mkdtemp(self) -> None:
        hf_body = self.src.split("def _retry_hf_dataset_with_compat_libs", 1)[1]
        self.assertNotIn("tempfile.mkdtemp", hf_body)
        self.assertNotIn("import tempfile", hf_body)


if __name__ == "__main__":
    unittest.main()
