"""Reliable AMD DevCloud Docker runner.

The Droplet cloud-init builds a local runtime image.  This module only uses
that local image; it never pulls an image as part of job execution.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import time

from dynamic_cloud.config import AmdSettings
from dynamic_cloud.runtime_layers import dataset_package_install_code
from dynamic_cloud.workspace import JobWorkspace


class RemoteHostRunner:
    """Runs layered ML payloads in the image built during Droplet cloud-init."""

    def __init__(self, settings: AmdSettings, remote_root: str | None = None):
        self.settings = settings
        if remote_root:
            self.remote_root = remote_root
        elif settings.admin_user == "root":
            self.remote_root = "/root/dynamic_jobs"
        else:
            self.remote_root = f"/home/{settings.admin_user}/dynamic_jobs"
        # Resolved once during the first bootstrap stage that needs torch.
        self._remote_python: str | None = None
        self.ssh_options = [
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "ControlMaster=auto",
            "-o",
            "ControlPath=/tmp/dc-%r@%h:%p",
            "-o",
            "ControlPersist=10m",
            "-o",
            "ConnectTimeout=15",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=20",
            "-o",
            "LogLevel=quiet",
            "-i",
            str(settings.ssh_private_key_path),
        ]

    # ── Upload ──────────────────────────────────────────────────────────

    def upload_workspace(self, ip: str, workspace: JobWorkspace, preserve_remote: bool = False) -> str:
        """Upload the local payload to the host via tar-over-SSH pipe.

        Streams the payload directory through SSH using a compressed tar pipe.
        This avoids SCP's fragile protocol handshake which breaks when the
        remote shell outputs MOTD/banner text on connection.

        Uses the exact same proven SSH handshake as ``ano_temp/run.py``:
        1. Wait for basic SSH login (echo 'Ready', 10s × 20 = 200s max)
        2. Wait for vendor provisioning to finish (echo 'SSHD_READY' and
           "Please wait" banner gone, 10s × 42 = 420s max)
        No cloud-init dependency — AMD DevCloud images use their own init.
        """
        self._wait_for_ssh_ready(ip)
        self._wait_for_provisioning(ip)
        self._silence_shell_banner(ip)

        remote_job_dir = f"{self.remote_root}/{workspace.job_id}"
        if preserve_remote:
            self._ssh(ip, f"mkdir -p {shlex.quote(remote_job_dir)}")
        else:
            self._ssh(ip, f"rm -rf {shlex.quote(remote_job_dir)} && mkdir -p {shlex.quote(remote_job_dir)}")

        print("Uploading job payload to the VM via SSH pipe...")
        self._upload_archive(ip, workspace, remote_job_dir)
        print("Job payload uploaded.")
        return remote_job_dir

    def _upload_archive(self, ip: str, workspace: JobWorkspace, remote_job_dir: str) -> None:
        """Upload via shell-based tar pipe, matching ano_temp's reliable pattern."""
        ssh_opts = " ".join(shlex.quote(o) for o in self.ssh_options)
        admin = shlex.quote(self.settings.admin_user)
        remote = shlex.quote(remote_job_dir)
        payload_dir = shlex.quote(str(workspace.payload_dir))

        tar_pipeline = (
            f"COPYFILE_DISABLE=1 tar -czf - -C {payload_dir} . 2>/dev/null "
            f"| ssh {ssh_opts} {admin}@{ip} "
            f"'mkdir -p {remote} && cd {remote} && tar -xzf - 2>/dev/null'"
        )
        result = subprocess.run(tar_pipeline, shell=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(
                f"Upload via SSH pipe failed (exit {result.returncode}). "
                f"The connection may have been interrupted during transfer."
            )

    # ── Dataset Inspection ──────────────────────────────────────────────

    def run_dataset_inspection(self, ip: str, remote_job_dir: str) -> None:
        """Prepare and inspect the dataset on the host natively."""
        self._wait_for_provisioning(ip)
        print("Downloading datasets to VM ......", flush=True)
        self._run_host_stage(ip, remote_job_dir, "dataset-inspect", "python3 -u dataset_inspector.py", install_packages=True)

    def download_dataset_metadata(self, ip: str, workspace: JobWorkspace, remote_job_dir: str) -> Path:
        """Retrieve and validate the dataset-inspection metadata.

        Uses SSH+Python to read the file — Python's ``open().read()`` is immune
        to banner text issues that affect ``cat`` (since the file path is not
        confused by banner output).  AMD DevCloud vendor images may prepend a
        welcome banner on stdout — the Python read approach avoids relying on
        parsing the banner out of ``cat`` output.
        """
        local_path = workspace.payload_dir / "dataset_metadata.json"
        file_path = remote_job_dir + "/dataset_metadata.json"
        safe_path = shlex.quote(file_path)

        # Debug: list the directory to confirm the file exists
        ls_result = self._ssh(
            ip,
            f"ls -la {shlex.quote(remote_job_dir)}/ 2>&1",
            capture=True, timeout_sec=30, print_output=False, check=False,
        )

        # Read via Python to avoid shell banner issues entirely.
        # Python's open() reads the file descriptor directly, so even if the
        # shell emits a banner, the file content is never mixed with it.
        try:
            result = self._ssh(
                ip,
                f"python3 -c 'import sys,json; print(json.dumps(json.load(open(sys.argv[1]))))' {safe_path} 2>&1",
                capture=True,
                timeout_sec=60,
                print_output=False,
                check=False,
            )
            raw = result.stdout or ""
        except BaseException as exc:
            raw = ""

        # Fallback: try base64 if Python read produced only banner text
        if not raw.strip() or "Please wait" in raw:
            b64_result = self._ssh(
                ip,
                f"python3 -c 'import base64; print(base64.b64encode(open({shlex.quote(file_path)},\"rb\").read()).decode())' 2>&1",
                capture=True,
                timeout_sec=60,
                print_output=False,
                check=False,
            )
            b64_raw = b64_result.stdout or ""
            # Strip banner text
            for line in b64_raw.splitlines():
                if "Please wait" not in line and line.strip():
                    try:
                        import base64
                        data = json.loads(base64.b64decode(line.strip()).decode("utf-8"))
                        local_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
                        print("Dataset metadata downloaded.")
                        return local_path
                    except Exception:
                        continue
            raise RuntimeError(
                f"Cannot read remote dataset_metadata.json. "
                f"ls output: {ls_result.stdout or '(empty)'}. "
                f"Raw python3 output: {raw!r}. "
                f"Raw base64 output: {b64_raw!r}."
            )

        # Strip banner lines from any remaining output
        clean_lines = [
            line for line in raw.splitlines()
            if "Please wait" not in line and line.strip()
        ]
        cleaned = "\n".join(clean_lines)

        if not cleaned:
            raise RuntimeError(
                f"No JSON found in Python output. "
                f"Raw output: {raw!r}. "
                f"Directory listing:\n{ls_result.stdout or '(empty)'}"
            )

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Remote dataset_metadata.json contains invalid JSON: {exc}.  "
                f"Content after banner stripping:\n{cleaned}"
            ) from exc

        local_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print("Dataset metadata downloaded.")
        return local_path

    # ── Training Execution ──────────────────────────────────────────────

    def _discover_remote_python(self, ip: str) -> str:
        """Find a Python interpreter on the remote host that can import ``torch``.

        AMD DevCloud PyTorch Quick Start images pre-install PyTorch but
        frequently inside a **conda environment** rather than the system
        ``python3``.  Non-interactive SSH commands do not source ``.bashrc`` /
        ``.profile`` so conda is never activated and ``import torch`` fails.

        This method probes the remote host in order of likelihood:

        1. System ``python3`` (fast path – works on images that install to
           the system Python).
        2. Conda environments found via ``conda env list``.
        3. Any ``python3`` binary under ``/opt`` that can import torch.

        The first match is cached on ``self._remote_python`` and returned.

        Returns the full Python path or command (e.g. ``"python3"``,
        ``"/opt/conda/envs/pytorch/bin/python3"``).
        """
        # Already discovered in this session
        if self._remote_python is not None:
            return self._remote_python

        print("  🔎 Probing remote host for a Python with torch...", flush=True)

        # ── Probe 0: common AMD Quick Start conda path ──
        # The PyTorch Quick Start image (amddevelopercloud-pytorch2100rocm724)
        # ships torch inside this conda environment — check it first before
        # running the slower discovery probes.
        for quick_path in (
            "/opt/conda/envs/pytorch/bin/python3",
            "/opt/conda/bin/python3",
        ):
            qr = self._ssh(ip, f"test -x {quick_path} && {quick_path} -c 'import torch; print(1)'",
                          capture=True, check=False, timeout_sec=15)
            if qr.returncode == 0:
                print(f"  ✅ Found torch at: {quick_path} (quick path)", flush=True)
                self._remote_python = quick_path
                return self._remote_python

        # ── Probe 1: system python3 ──
        r = self._ssh(ip, "python3 -c 'import torch; print(1)'",
                       capture=True, check=False, timeout_sec=15)
        if r.returncode == 0:
            print(f"  ✅ Using system python3 (torch available)", flush=True)
            self._remote_python = "python3"
            return self._remote_python

        # ── Probe 2: conda environments ──
        r = self._ssh(ip,
            "which conda 2>/dev/null || "
            "for d in /opt/conda /opt/miniconda3 /opt/miniconda "
            "/root/miniconda3 /root/anaconda3 /root/anaconda "
            "/usr/local/conda; do "
            "  p=\"$d/bin/conda\"; "
            "  [ -x \"$p\" ] && echo \"$p\" && break; "
            "done 2>/dev/null",
            capture=True, check=False, timeout_sec=15)
        conda_bin = r.stdout.strip() if r.returncode == 0 else ""
        if conda_bin:
            # Get the conda base directory
            base_r = self._ssh(ip,
                f"{conda_bin} info --base 2>/dev/null",
                capture=True, check=False, timeout_sec=15)
            conda_base = base_r.stdout.strip() if base_r.returncode == 0 else ""
            if conda_base:
                # List envs and probe each
                envs_r = self._ssh(ip,
                    f"{conda_bin} env list 2>/dev/null | grep -v '^#' | "
                    f"awk '{{print $NF}}' | head -20",
                    capture=True, check=False, timeout_sec=15)
                env_paths = [ln.strip() for ln in envs_r.stdout.strip().split('\n')
                             if ln.strip()] if envs_r.returncode == 0 else []

                for env_path in env_paths:
                    if env_path == "base" or env_path == conda_base:
                        py = f"{conda_base}/bin/python3"
                    elif env_path.startswith("/"):
                        py = f"{env_path}/bin/python3"
                    else:
                        py = f"{conda_base}/envs/{env_path}/bin/python3"

                    env_r = self._ssh(ip, f"test -x {py} && {py} -c 'import torch; print(1)'",
                                      capture=True, check=False, timeout_sec=15)
                    if env_r.returncode == 0:
                        print(f"  ✅ Found torch in conda env: {py}", flush=True)
                        self._remote_python = py
                        return self._remote_python

        # ── Probe 3: brute force python3 binaries ──
        find_r = self._ssh(ip,
            "find /opt /usr/local /root -name python3 -type f "
            "2>/dev/null | head -20",
            capture=True, check=False, timeout_sec=30)
        if find_r.returncode == 0:
            for py_path in [ln.strip() for ln in find_r.stdout.strip().split('\n') if ln.strip()]:
                probe_r = self._ssh(ip, f"{py_path} -c 'import torch; print(1)'",
                                    capture=True, check=False, timeout_sec=15)
                if probe_r.returncode == 0:
                    print(f"  ✅ Found torch at: {py_path}", flush=True)
                    self._remote_python = py_path
                    return self._remote_python

        # ── Probe 4: search for torch module directly ──
        torch_r = self._ssh(ip,
            "find /usr /opt /root /home -path '*/site-packages/torch/__init__.py' "
            "-not -path '*/proc/*' -type f 2>/dev/null | head -5",
            capture=True, check=False, timeout_sec=15)
        if torch_r.returncode == 0:
            torch_paths = [ln.strip() for ln in torch_r.stdout.strip().split('\n') if ln.strip()]
            for tp in torch_paths:
                # Derive python from the site-packages path.
                maybe_py = tp.replace("/lib/python", "/bin/python3").replace(
                    "/site-packages/torch/__init__.py", "")
                for candidate in (maybe_py, maybe_py + ".12", maybe_py + ".11",
                                   f"{'/'.join(tp.split('/')[:-4])}/bin/python3"):
                    probe_r = self._ssh(ip,
                        f"test -x {candidate} && {candidate} -c 'import torch; print(1)'",
                        capture=True, check=False, timeout_sec=15)
                    if probe_r.returncode == 0:
                        print(f"  ✅ Found torch at: {candidate} (via torch module)", flush=True)
                        self._remote_python = candidate
                        return self._remote_python

        # ── Fallback: cache and return default (will fail gracefully later) ──
        print("  ⚠️  No Python with torch found; will install torch overlay", flush=True)
        self._remote_python = "python3"
        return self._remote_python

    def execute_bootstrap(self, ip: str, remote_job_dir: str) -> None:
        """Run the generated workload directly on the host natively."""
        self._wait_for_provisioning(ip)
        python_cmd = self._discover_remote_python(ip)
        qdir = shlex.quote(remote_job_dir)
        overlay = '/opt/dynamic-cloud/python-packages'

        # If we fell back to system python3 without torch, install ROCm-compatible
        # PyTorch from the AMD/PyTorch index into the overlay.
        # NOTE: PyTorch's wheel index uses major.minor version (e.g. rocm7.2),
        # NOT major.minor.patch (rocm7.2.4).
        if python_cmd == "python3":
            check = self._ssh(ip,
                f"export PYTHONPATH={qdir}:{overlay} && "
                f"python3 -c 'import torch; print(1)'",
                capture=True, check=False, timeout_sec=15)
            if check.returncode != 0:
                print("  📦 Installing ROCm PyTorch to package overlay...", flush=True)
                # Pre-flight: verify the ROCm wheel index is reachable and
                # has a wheel for the current Python version before attempting
                # a large install that will just fail.
                py_ver_check = self._ssh(ip,
                    "python3 -c 'import sys; print(f\"{sys.version_info.major}.{sys.version_info.minor}\")'",
                    capture=True, check=False, timeout_sec=15)
                rocm_variants = ("rocm7.2", "rocm6.2")
                install_ok = False
                last_error = ""
                for rocm_variant in rocm_variants:
                    index_url = f"https://download.pytorch.org/whl/{rocm_variant}"
                    # Quick check: can we reach the index at all?
                    probe = self._ssh(ip,
                        f"python3 -c 'import urllib.request; "
                        f"r = urllib.request.urlopen(\"{index_url}/torch/\", timeout=15); "
                        f"print(r.status)' 2>&1",
                        capture=True, check=False, timeout_sec=20)
                    if probe.returncode != 0 or "200" not in (probe.stdout or ""):
                        last_error = f"index {index_url} not reachable (status: {probe.stdout.strip() if probe.stdout else 'timeout'})"
                        print(f"  ⚠️  {last_error}; trying next variant...", flush=True)
                        continue
                    try:
                        install_torch = (
                            f"mkdir -p {overlay} && "
                            f"export PYTHONPATH={qdir}:{overlay} && "
                            f"pip install torch torchvision torchaudio "
                            f"--target {overlay} "
                            f"--index-url {index_url} "
                            f"--break-system-packages 2>&1"
                        )
                        self._ssh(ip, install_torch, capture=True, check=True, timeout_sec=1200)
                        # Verify the install actually works
                        py_ver = self._ssh(ip,
                            f"export PYTHONPATH={qdir}:{overlay} && "
                            f"python3 -c 'import torch; print(torch.__version__)'",
                            capture=True, check=True, timeout_sec=15)
                        print(f"  ✅ ROCm PyTorch {py_ver.stdout.strip()} installed in overlay "
                              f"(via {rocm_variant})", flush=True)
                        install_ok = True
                        break
                    except RuntimeError as exc:
                        last_error = str(exc)
                        print(f"  ⚠️  Install failed via {rocm_variant}: {last_error}", flush=True)
                        continue

                if not install_ok:
                    raise RuntimeError(
                        f"Could not install ROCm PyTorch on the AMD VM. "
                        f"Checked indices: {', '.join(rocm_variants)}. "
                        f"Last error: {last_error}. "
                        f"The VM's Python may not be compatible with the available ROCm wheels. "
                        f"Try using the PyTorch Quick Start image (amddevelopercloud-pytorch2100rocm724) "
                        f"which has torch pre-installed."
                    )

        self._run_host_stage(ip, remote_job_dir, "runtime",
                             f"{python_cmd} -u runtime_bootstrap.py",
                             install_packages=False)
        print("Bootstrap execution completed.")

    # ── Output Download ─────────────────────────────────────────────────

    def download_outputs(self, ip: str, workspace: JobWorkspace, remote_job_dir: str) -> Path:
        """Download the remote ``outputs/`` directory into the run folder.

        The transfer is an **atomic, size-verified file copy** rather than a
        live ``tar | ssh`` pipe:

        1. Pack ``outputs/`` into a gzip tarball **file on the VM**
           (``tar -czf <file>``), then read back its byte size with ``stat``.
           Writing to a regular file — not stdout — means there is no SSH
           stdout pipe for a lingering VM background process to hold open, so
           the remote command always returns a clean exit code.
        2. ``scp`` the single tarball down to a local temp file. ``scp`` is
           atomic, integrity-checked end-to-end, and reuses the SSH retry logic
           in ``_run`` for transient network errors.
        3. Verify the local file size matches the remote ``stat`` size — a
           mismatch means the transfer was truncated and we raise immediately
           (no silent partial download).
        4. Extract the tarball into ``workspace.outputs_dir.parent`` (the
           per-execution run folder) so artifacts land at
           ``<run_dir>/outputs/...`` with no post-run shift, then remove the
           tarball and clean up the remote copy.

        Remote tar stderr is redirected to a file on the VM (not piped through
        SSH) so a verbose tar can never block the command; we read it back only
        if the pack step fails.
        """
        import shutil as _shutil

        outputs_dir = workspace.outputs_dir
        # Clean stale outputs from any previous run in this run folder, then
        # (re)create the directory so the extraction target exists.
        if outputs_dir.exists():
            _shutil.rmtree(outputs_dir)
        outputs_dir.parent.mkdir(parents=True, exist_ok=True)
        outputs_dir.mkdir(parents=True, exist_ok=True)
        extract_dir = outputs_dir.parent

        def _format_size(byte_count: int) -> str:
            """Human-readable size — KB for < 1 MB, MB otherwise."""
            if byte_count < 1024 * 1024:
                return f"{byte_count / 1024:.1f} KB"
            return f"{byte_count / (1024 * 1024):.1f} MB"

        remote_tarball = f"{remote_job_dir}/outputs.tar.gz"
        remote_err_file = "/tmp/dc_tar_pack.err"
        download_start = time.monotonic()

        # ── 1. Pack outputs/ into a tarball file on the VM + read its size ──
        print("  ⬇  Packing outputs on the VM…", flush=True)
        pack_cmd = (
            f"cd {shlex.quote(remote_job_dir)} && "
            f"rm -f {shlex.quote(remote_tarball)} && "
            f"timeout 1800 tar -czf {shlex.quote(remote_tarball)} "
            f"--exclude-caches-all outputs 2>{shlex.quote(remote_err_file)} && "
            f"stat -c %s {shlex.quote(remote_tarball)}"
        )
        pack_result = self._ssh(
            ip, pack_cmd, capture=True, check=False,
            timeout_sec=1800, print_output=False,
        )
        if pack_result.returncode != 0:
            tar_stderr = ""
            try:
                err_result = self._ssh(
                    ip,
                    f"cat {shlex.quote(remote_err_file)} 2>/dev/null || true",
                    capture=True, check=False, timeout_sec=30, print_output=False,
                )
                tar_stderr = (err_result.stdout or "").strip()
            except Exception:
                pass
            raise RuntimeError(
                f"Remote tar of outputs failed (exit {pack_result.returncode}): "
                f"{tar_stderr or pack_result.stderr or 'No stderr was returned.'}"
            )
        size_lines = (pack_result.stdout or "").strip().splitlines()
        if not size_lines:
            raise RuntimeError(
                "Remote tar succeeded but did not report a tarball size "
                "(empty `stat` output). Cannot verify the download."
            )
        try:
            remote_size = int(size_lines[-1].strip())
        except ValueError as exc:
            raise RuntimeError(
                f"Could not parse remote tarball size from `stat` output "
                f"{size_lines[-1]!r}: {exc}"
            )

        # ── 2. scp the tarball down to a local temp file ──
        local_tarball = extract_dir / "outputs.tar.gz.part"
        print(
            f"  ⬇  Downloading outputs… {_format_size(remote_size)}",
            flush=True,
        )
        self._run(
            [
                "scp", *self.ssh_options,
                f"{self.settings.admin_user}@{ip}:{remote_tarball}",
                str(local_tarball),
            ],
            capture=True, print_output=False, timeout_sec=1800,
        )

        # ── 3. Verify the local size matches the remote size ──
        if not local_tarball.exists():
            raise RuntimeError(
                f"scp reported success but the tarball is missing at "
                f"{local_tarball}."
            )
        local_size = local_tarball.stat().st_size
        if local_size != remote_size:
            raise RuntimeError(
                f"Downloaded tarball is incomplete: local size {local_size} "
                f"bytes != remote size {remote_size} bytes. The transfer was "
                f"truncated — not extracting a partial archive."
            )

        # ── 4. Extract locally, then remove the tarball ──
        extract = subprocess.run(
            ["tar", "-xzf", str(local_tarball), "-C", str(extract_dir)],
            capture_output=True, text=True, timeout=600,
        )
        if extract.returncode != 0:
            raise RuntimeError(
                f"Local extraction of the downloaded tarball failed "
                f"(exit {extract.returncode}): {extract.stderr.strip()}"
            )
        try:
            local_tarball.unlink()
        except OSError:
            pass

        # Best-effort cleanup of the remote tarball + stderr file.
        try:
            self._ssh(
                ip,
                f"rm -f {shlex.quote(remote_tarball)} {shlex.quote(remote_err_file)}",
                capture=True, check=False, timeout_sec=30, print_output=False,
            )
        except Exception:
            pass

        elapsed = time.monotonic() - download_start
        print(
            f"  ✅ Download complete: {_format_size(remote_size)} "
            f"in {elapsed:.0f}s",
            flush=True,
        )
        print(f"Outputs downloaded to {outputs_dir}")
        for output in sorted(os.listdir(outputs_dir)):
            print(f"  {output}")
        return outputs_dir

    def _silence_shell_banner(self, ip: str) -> None:
        """Suppress vendor banners for non-interactive SSH connections.

        AMD DevCloud images may print "Please wait while we get your droplet
        ready..." from profile.d scripts on every SSH command.  This function
        uses only **non-destructive** techniques that never touch /etc/ files:

        1. Creates ``~/.hushlogin`` — tells sshd to skip the MOTD banner.
        2. Backs up any existing ``~/.bashrc`` to ``~/.bashrc.dc_backup`` (once,
           idempotently) and writes a new ``~/.bashrc`` that exits early for
           non-interactive shells (preventing profile.d banner scripts from
           running during our command-only SSH sessions) while still sourcing
           the user's *original* bashrc for interactive logins.

        No system files are modified; no chmod is applied; the user's original
        bashrc is preserved (not clobbered).
        """
        safe_bashrc = (
            r'# Dynamic Cloud — suppress banner for non-interactive SSH commands.'
            r'\n# Non-interactive shells return early so profile.d banner scripts'
            r'\n# never run during our command-only SSH sessions.'
            r'\nif [[ $- != *i* ]]; then return; fi'
            r'\n# Interactive shell: restore the original bashrc (backed'
            r'\n# up once to ~/.bashrc.dc_backup before we overwrote it).'
            r'\nif [ -f ~/.bashrc.dc_backup ]; then . ~/.bashrc.dc_backup; fi'
        )
        cmd = (
            'export PATH=/usr/sbin:/usr/bin:/sbin:/bin; '
            # Back up the user's original ~/.bashrc exactly once, BEFORE we
            # overwrite it, so interactive shells can still source it.
            'if [ -f ~/.bashrc ] && [ ! -f ~/.bashrc.dc_backup ]; then '
            "cp ~/.bashrc ~/.bashrc.dc_backup; "
            'fi; '
            r"touch ~/.hushlogin 2>/dev/null; "
            r"printf '%b\n' " + shlex.quote(safe_bashrc) + r" > ~/.bashrc; "
            r':'
        )
        self._ssh(ip, cmd, print_output=False, check=False, timeout_sec=30)

    def _run_host_stage(self, ip: str, remote_job_dir: str, suffix: str,
                         command: str, install_packages: bool) -> None:
        qdir = shlex.quote(remote_job_dir)
        cache_dir = shlex.quote(self._remote_dataset_cache_dir())

        # Ensure outputs and dataset cache exist, and symlink cache into workspace
        setup_cmd = (
            f"mkdir -p {qdir}/outputs {cache_dir} && "
            f"ln -sfn {cache_dir} {qdir}/prepared_datasets"
        )
        self._ssh(ip, setup_cmd, print_output=False)

        overlay = '/opt/dynamic-cloud/python-packages'
        # Install packages natively into overlay (system python3 is fine here
        # since packages go to --target, not site-packages).
        if install_packages:
            pkg_cmd = (
                f"export DYNAMIC_CLOUD_WORKSPACE={qdir} && "
                f"export PYTHONPATH={qdir}:{overlay} && "
                f"python3 -c {shlex.quote(dataset_package_install_code())}"
            )
            self._ssh(ip, pkg_cmd, capture=False, timeout_sec=3600)

        # Execute payload using the discovered Python (may be a conda path).
        runtime = (
            f"export DYNAMIC_CLOUD_WORKSPACE={qdir} && "
            f"export DYNAMIC_CLOUD_OUTPUTS_DIR={qdir}/outputs && "
            f"export PYTHONPATH={qdir}:{overlay} && "
            f"cd {qdir} && "
            f"mkdir -p outputs/logs && "
            f"set -o pipefail && "
            f"{command} 2>&1 | tee outputs/logs/{suffix}.log"
        )
        self._ssh(ip, runtime, capture=False, timeout_sec=86400)

    def _remote_dataset_cache_dir(self) -> str:
        """Return the VM-level cache shared by inspection and training stages."""
        return f"{self.remote_root}/datasets"

    # ── SSH Utilities ───────────────────────────────────────────────────

    def _ssh(self, ip: str, remote_cmd: str, **kwargs) -> subprocess.CompletedProcess[str]:
        return self._run(
            ["ssh", *self.ssh_options, f"{self.settings.admin_user}@{ip}", remote_cmd],
            **kwargs,
        )

    def _ssh_probe_options(self) -> list[str]:
        """SSH options used for initial probing (no ControlMaster).

        ControlMaster can interfere when the remote host is still booting or
        its host key changes during AMD DevCloud provisioning.  These options
        treat each probe as a fresh connection.
        """
        return [
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=15",
            "-i", str(self.settings.ssh_private_key_path),
        ]

    def _wait_for_ssh_ready(self, ip: str) -> None:
        """Wait for basic SSH key login to succeed.

        Polls with ``echo 'Ready'`` every 10 seconds until the timeout is hit.
        Configurable via ``AMD_SSH_WAIT_TIMEOUT_SEC`` (default 600s).

        Raises ``TimeoutError`` if SSH never becomes available.
        """
        timeout_sec = max(self.settings.wait_ssh_timeout_sec, 60)
        max_attempts = timeout_sec // 10
        print(f"🔒 Probing VM for SSH handshake availability (timeout {timeout_sec}s)...")
        cmd = [
            "ssh", *self._ssh_probe_options(), f"{self.settings.admin_user}@{ip}",
            "echo 'Ready'",
        ]
        for _ in range(max_attempts):
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            except subprocess.TimeoutExpired:
                result = subprocess.CompletedProcess(cmd, returncode=255, stdout="", stderr="timeout")
            if result.returncode == 0 and "Ready" in result.stdout:
                print("\n⚡ SSH service is alive and accepting keys!")
                return
            print("⏳", end="", flush=True)
            time.sleep(10)
        raise TimeoutError(
            f"SSH login for {self.settings.admin_user}@{ip} did not become ready "
            f"within {timeout_sec} seconds. Verify AMD_SSH_PRIVATE_KEY_PATH matches the "
            f"key attached to the Droplet."
        )

    def _wait_for_provisioning(self, ip: str) -> None:
        """Wait out AMD vendor initialization banner and reboots.

        Polls with ``echo 'SSHD_READY'`` every 10 seconds until the timeout.
        Configurable via ``AMD_PROVISIONING_WAIT_TIMEOUT_SEC`` (default 600s).
        We know provisioning is done when:
        - stdout contains "SSHD_READY" (SSH is responsive)
        - output does NOT contain "Please wait" (vendor banner is gone)
        """
        timeout_sec = max(self.settings.wait_provisioning_timeout_sec, 60)
        max_attempts = timeout_sec // 10
        print(f"🛠️ Waiting for AMD system initialization (timeout {timeout_sec}s)...")
        cmd = [
            "ssh", *self._ssh_probe_options(), f"{self.settings.admin_user}@{ip}",
            "echo 'SSHD_READY'",
        ]
        for _ in range(max_attempts):
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            except subprocess.TimeoutExpired:
                print("🔄", end="", flush=True)
                time.sleep(10)
                continue
            output = result.stdout + result.stderr

            if "SSHD_READY" in result.stdout and "Please wait" not in output:
                print("\n✨ AMD environment is completely stable and ready!")
                return

            if "Connection refused" in output:
                print("🔄", end="", flush=True)
            elif "Please wait" in output:
                print("⏳", end="", flush=True)
            else:
                print(".", end="", flush=True)

            time.sleep(10)

        raise TimeoutError(
            f"AMD environment initialization at {ip} did not complete within "
            f"{timeout_sec} seconds. The vendor banner may still be active or SSH "
            f"may have failed to stabilize. Increase AMD_PROVISIONING_WAIT_TIMEOUT_SEC "
            f"if the image needs more time."
        )

    @staticmethod
    def _command_stderr(result: subprocess.CompletedProcess) -> str:
        stderr = result.stderr or ""
        return stderr.decode(errors="replace") if isinstance(stderr, bytes) else str(stderr)

    @staticmethod
    def _ssh_retry_count() -> int:
        raw_value = os.getenv("DYNAMIC_CLOUD_SSH_RETRIES", "5").strip()
        try:
            return max(1, int(raw_value))
        except ValueError:
            return 5

    @staticmethod
    def _run(
        cmd: list[str],
        capture: bool = True,
        timeout_sec: int | None = None,
        check: bool = True,
        print_output: bool = True,
        input_text: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        max_attempts = RemoteHostRunner._max_ssh_attempts(cmd, capture)
        if os.getenv("DYNAMIC_CLOUD_DEBUG_COMMANDS") == "1":
            print(f"Running: {' '.join(RemoteHostRunner._redact_command(cmd))}")
        for attempt in range(1, max_attempts + 1):
            # Always capture stderr for SSH commands so _is_transient_ssh_error
            # can detect "Connection refused" etc. even when stdout goes to
            # the terminal (capture=False for live progress output).
            if capture or cmd[0] == "ssh":
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE if capture else None,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=timeout_sec,
                    input=input_text,
                )
            else:
                result = subprocess.run(cmd, capture_output=False, text=True, timeout=timeout_sec, input=input_text)

            if not (check and result.returncode != 0):
                if capture and print_output and result.stdout:
                    print(result.stdout)
                return result

            if attempt < max_attempts and RemoteHostRunner._is_transient_ssh_error(result):
                delay = min(5 * attempt, 30)
                print(
                    f"SSH connection was temporarily unavailable; retrying "
                    f"in {delay}s ({attempt}/{max_attempts})..."
                )
                time.sleep(delay)
                continue

            stderr = result.stderr or ""
            if not capture and stderr:
                print(stderr)
            elif capture and result.stderr:
                print(result.stderr)
            stdout = result.stdout or ""
            detail_parts = [f"exit code {result.returncode}"]
            if stdout:
                detail_parts.append(f"\n[REMOTE STDOUT]\n{stdout.strip()[-2000:]}")
            if stderr:
                detail_parts.append(f"\n[REMOTE STDERR]\n{stderr.strip()[-2000:]}")
            raise RuntimeError(
                f"Command failed: {' '.join(RemoteHostRunner._redact_command(cmd))}\n"
                + "\n".join(detail_parts)
            )

        return result

    @staticmethod
    def _max_ssh_attempts(cmd: list[str], capture: bool) -> int:
        if not cmd or cmd[0] not in {"ssh", "scp"}:
            return 1
        return RemoteHostRunner._ssh_retry_count()

    @staticmethod
    def _is_transient_ssh_error(result: subprocess.CompletedProcess[str]) -> bool:
        if result.returncode != 255:
            return False
        stderr = RemoteHostRunner._command_stderr(result).lower()
        transient_markers = (
            "connection refused",
            "connection reset",
            "connection timed out",
            "operation timed out",
            "no route to host",
            "network is unreachable",
            "connection closed",
            "connection to",
            "broken pipe",
        )
        return any(marker in stderr for marker in transient_markers)

    @staticmethod
    def _redact_command(cmd: list[str]) -> list[str]:
        redacted = list(cmd)
        for index, part in enumerate(redacted[:-1]):
            if part == "-i":
                redacted[index + 1] = "<SSH_KEY>"
        return redacted