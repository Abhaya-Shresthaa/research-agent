from __future__ import annotations

import base64
import hashlib
import json
import socket
import subprocess
import time
from typing import Any

import requests

from dynamic_cloud.config import AmdSettings


class AmdDropletManager:
    """Manages the full lifecycle of an AMD DevCloud GPU Droplet.

    Lifecycle: create/adopt → wait_until_active → get_public_ip → (use) → destroy.
    GPU Droplets bill per-second with a 5-minute minimum. Powered-off Droplets
    still incur charges for reserved GPU resources; destroying is the only way
    to stop billing.
    """

    BASE_URL = "https://api.devcloud.amd.com/v2"
    FALLBACK_BASE_URL = "https://api-amd.digitalocean.com/v2"

    def __init__(self, settings: AmdSettings):
        self.settings = settings
        self.headers = {
            "Authorization": f"Bearer {settings.api_key}",
            "Content-Type": "application/json",
        }
        self.base_url = settings.api_base_url.rstrip("/") or self.BASE_URL
        self.droplet_id: int | None = None
        self.region: str = settings.region

    def adopt(self, droplet_id: int) -> None:
        """Attach this manager to an existing Droplet ID for cleanup/reuse."""
        self.droplet_id = droplet_id

    # ── SSH Key Management ──────────────────────────────────────────────

    def ensure_ssh_key(self) -> str | int:
        """Ensure the SSH public key is registered on the DevCloud account.

        Returns the key fingerprint or ID for use in Droplet creation.
        """
        explicit_keys = tuple(self.settings.ssh_keys)
        if explicit_keys:
            return explicit_keys[0]

        if self.settings.ssh_key_name:
            key_id = self._find_ssh_key(self.settings.ssh_key_name)
            if key_id:
                return key_id
            if not self.settings.register_ssh_key and not self.settings.ssh_public_key:
                raise RuntimeError(
                    f"SSH key {self.settings.ssh_key_name!r} was not found. "
                    "Set AMD_SSH_PUBLIC_KEY to create it, set AMD_REGISTER_SSH_KEY=1, "
                    "or set AMD_SSH_KEYS to an existing key ID/name."
                )

        if not self.settings.register_ssh_key and not self.settings.ssh_public_key:
            raise RuntimeError(
                "SSH key registration is disabled. Set AMD_SSH_KEYS to an existing DevCloud key "
                "or set AMD_SSH_PUBLIC_KEY to let this workflow find/register the key."
            )

        if self.settings.ssh_public_key:
            pub_key = self.settings.ssh_public_key
        elif self.settings.ssh_public_key_path.exists():
            pub_key = self.settings.ssh_public_key_path.read_text(encoding="utf-8").strip()
        else:
            raise FileNotFoundError(
                f"SSH public key not found at {self.settings.ssh_public_key_path}. "
                "Set AMD_SSH_PUBLIC_KEY directly, or generate one with: ssh-keygen -t rsa -b 4096"
            )

        existing = self._find_ssh_key_by_public_key(pub_key)
        if existing:
            print("SSH key already registered; reusing existing key ID.")
            return existing

        key_name = self.settings.ssh_key_name or "ml-orchestrator-auto"
        print(f"Registering SSH key '{key_name}' on AMD DevCloud...")
        resp = self._request(
            "post",
            f"{self.base_url}/account/keys",
            json={"name": key_name, "public_key": pub_key},
        )
        resp.raise_for_status()
        key_data = resp.json()["ssh_key"]
        print(f"SSH key registered (ID: {key_data['id']}, fingerprint: {key_data['fingerprint']}).")
        return key_data["id"]

    def _find_ssh_key(self, name_or_fingerprint: str) -> int | None:
        """Look up an SSH key by name or fingerprint. Returns key ID or None."""
        for key in self._list_ssh_keys():
            if (key.get("name") == name_or_fingerprint
                    or key.get("fingerprint") == name_or_fingerprint
                    or str(key.get("id")) == name_or_fingerprint):
                return key["id"]
        return None

    def _find_ssh_key_by_public_key(self, pub_key: str) -> int | None:
        wanted = self._normalize_public_key(pub_key)
        wanted_fingerprint = self._public_key_md5_fingerprint(pub_key)
        for key in self._list_ssh_keys():
            existing_public_key = str(key.get("public_key") or "")
            try:
                existing = self._normalize_public_key(existing_public_key) if existing_public_key else ""
            except ValueError:
                existing = ""
            existing_fingerprint = str(key.get("fingerprint") or "")
            if existing == wanted or (wanted_fingerprint and existing_fingerprint == wanted_fingerprint):
                return key["id"]
        return None

    def _list_ssh_keys(self) -> list[dict[str, Any]]:
        keys: list[dict[str, Any]] = []
        page = 1
        while True:
            resp = self._request(
                "get",
                f"{self.base_url}/account/keys",
                params={"per_page": 200, "page": page},
            )
            if resp.status_code == 404 and self.base_url != self.FALLBACK_BASE_URL:
                self.base_url = self.FALLBACK_BASE_URL
                print(f"AMD key API route not found; retrying with {self.base_url}.")
                keys.clear()
                page = 1
                continue
            resp.raise_for_status()
            payload = resp.json()
            keys.extend(payload.get("ssh_keys", []))
            links = payload.get("links", {}).get("pages", {})
            if not links.get("next"):
                break
            page += 1
        return keys

    @staticmethod
    def _normalize_public_key(pub_key: str) -> str:
        parts = pub_key.split()
        if len(parts) < 2:
            raise ValueError("Invalid SSH public key format.")
        base64.b64decode(parts[1])
        return " ".join(parts[:2])

    @staticmethod
    def _public_key_md5_fingerprint(pub_key: str) -> str | None:
        parts = pub_key.split()
        if len(parts) < 2:
            return None
        try:
            key_bytes = base64.b64decode(parts[1])
        except Exception:
            return None
        digest = hashlib.md5(key_bytes).hexdigest()
        return ":".join(digest[index:index + 2] for index in range(0, len(digest), 2))

    # ── Droplet Creation ────────────────────────────────────────────────

    def create_droplet(self) -> None:
        """Create a fresh GPU Droplet. Always creates new — no reuse.

        Tries each size in ``self.settings.vm_sizes`` until one succeeds,
        similar to the original multi-size fallback pattern.
        """
        ssh_keys = self._droplet_ssh_keys()
        droplet_name = self.settings.vm_name

        image_error: str | None = None
        plan_errors: list[str] = []
        regions = self.settings.regions or (self.settings.region,)
        for region in regions:
            for size in self.settings.vm_sizes:
                short_size = size.replace("gpu-", "").replace("devcloud", "").strip("-")
                print(
                    f"  ▸ Creating GPU droplet — {short_size} · image: {self.settings.gpu_image}"
                )
                try:
                    resp = self._request(
                        "post",
                        f"{self.base_url}/droplets",
                        json=self._create_droplet_payload(droplet_name, region, size, ssh_keys),
                    )
                    if resp.status_code == 404 and self.base_url != self.FALLBACK_BASE_URL:
                        self.base_url = self.FALLBACK_BASE_URL
                        print(f"AMD Droplet API route not found; retrying with {self.base_url}.")
                        return self.create_droplet()
                    if resp.status_code == 422:
                        error_msg = resp.json().get("message", resp.text)
                        lower_error = str(error_msg).lower()
                        if "invalid image" in lower_error or "image" in lower_error:
                            image_error = str(error_msg)
                            print(f"  image {self.settings.gpu_image} cannot be used in {region}: {error_msg}")
                            break
                        plan_errors.append(f"{region}/{size}: {error_msg}")
                        print(f"  {size} not available in {region}: {error_msg}")
                        continue
                    resp.raise_for_status()
                    response_json = resp.json()
                    if not isinstance(response_json, dict):
                        raise RuntimeError(
                            f"Unexpected API response type: expected JSON object, got {type(response_json).__name__}. "
                            f"Response: {resp.text[:2000]}"
                        )
                    if "droplet" not in response_json:
                        raise RuntimeError(
                            f"AMD DevCloud API returned a successful status ({resp.status_code}) but the response "
                            f"does not contain a 'droplet' key. This usually means the image slug, size slug, "
                            f"or another parameter is invalid.\n"
                            f"  Image: {self.settings.gpu_image!r}\n"
                            f"  Size: {size!r}\n"
                            f"  Region: {region!r}\n"
                            f"  API Response: {json.dumps(response_json, indent=2)[:2000]}"
                        )
                    droplet_data = response_json["droplet"]
                    self.droplet_id = droplet_data["id"]
                    self.region = region
                    print(f"  ✓ Droplet created (ID: {self.droplet_id})")
                    self._assign_project_best_effort()
                    return
                except requests.exceptions.HTTPError as exc:
                    if exc.response is not None and exc.response.status_code == 422:
                        plan_errors.append(f"{region}/{size}: {exc}")
                        print(f"  {size} cannot be used in {region}: {exc}")
                        continue
                    raise

        raise RuntimeError(
            (
                f"No usable AMD GPU Droplet image found. AMD DevCloud rejected "
                f"AMD_GPU_IMAGE={self.settings.gpu_image!r} in regions {', '.join(regions)}: {image_error}. "
                "Use an account-visible AMD Developer Cloud image such as "
                "amddevelopercloud-pytorch2100rocm724 for PyTorch or amddeveloperclou-rocm724 for ROCm."
            )
            if image_error and not plan_errors
            else (
                "No usable GPU Droplet plan found. "
                "Check AMD_REGIONS/AMD_REGION supports GPU Droplets and your account has GPU credits. "
                "If AMD DevCloud UI can create this MI300X Droplet but the API cannot, "
                "create it in the UI and rerun with --amd-droplet-id <id>. "
                f"API errors: {'; '.join(plan_errors) if plan_errors else 'none'}"
            )
        )

    def _assign_project_best_effort(self) -> None:
        if not self.settings.project_id or not self.droplet_id:
            return

        urn = f"do:droplet:{self.droplet_id}"
        try:
            resp = self._request(
                "post",
                f"{self.base_url}/projects/{self.settings.project_id}/resources",
                json={"resources": [urn]},
            )
            resp.raise_for_status()
            print(f"Assigned Droplet {self.droplet_id} to project {self.settings.project_id}.")
        except requests.exceptions.RequestException as exc:
            print(
                f"Warning: could not assign Droplet {self.droplet_id} "
                f"to project {self.settings.project_id}: {exc}"
            )

    # ── Wait for Active ─────────────────────────────────────────────────

    def wait_until_active(self) -> None:
        """Poll ``GET /v2/droplets/{id}`` until status is ``active``, then wait for SSH.

        GPU Droplets typically take 2-5 minutes to provision.
        """
        if not self.droplet_id:
            raise RuntimeError("No Droplet ID. Call create_droplet() first.")

        print("Waiting for Droplet to become active...")
        deadline = time.monotonic() + self.settings.wait_droplet_timeout_sec
        while time.monotonic() < deadline:
            resp = self._request(
                "get",
                f"{self.base_url}/droplets/{self.droplet_id}",
            )
            resp.raise_for_status()
            droplet = resp.json()["droplet"]
            status = droplet["status"]
            if status == "active":
                print("Droplet is active. Waiting for SSH...")
                ip = self.get_public_ip()
                self.wait_for_ssh(ip)
                return
            if status == "errored":
                raise RuntimeError(
                    f"Droplet {self.droplet_id} entered error state during provisioning."
                )
            time.sleep(10)

        raise TimeoutError(
            f"Droplet {self.droplet_id} did not become active within "
            f"{self.settings.wait_droplet_timeout_sec} seconds."
        )

    # ── Public IP ───────────────────────────────────────────────────────

    def get_public_ip(self) -> str:
        """Extract the public IPv4 address from the Droplet's network info."""
        if not self.droplet_id:
            raise RuntimeError("No Droplet ID.")

        resp = self._request(
            "get",
            f"{self.base_url}/droplets/{self.droplet_id}",
        )
        resp.raise_for_status()
        droplet = resp.json()["droplet"]
        for net in droplet.get("networks", {}).get("v4", []):
            if net["type"] == "public":
                return net["ip_address"]
        raise RuntimeError(f"Droplet {self.droplet_id} has no public IPv4 address.")

    # ── Destroy ─────────────────────────────────────────────────────────

    def destroy(self) -> None:
        """Destroy the Droplet. Always destroy — never deallocate/keep.

        GPU Droplets bill per-second with 5-minute minimum.
        Destroying is the ONLY way to stop billing.
        """
        if not self.droplet_id:
            print("No Droplet to destroy.")
            return

        print(f"Destroying Droplet {self.droplet_id}...")
        for attempt in range(1, 4):
            try:
                resp = self._request(
                    "delete",
                    f"{self.base_url}/droplets/{self.droplet_id}",
                )
                if resp.status_code in {200, 204}:
                    print(f"Droplet {self.droplet_id} destroyed.")
                    self.droplet_id = None
                    return
                if resp.status_code == 404:
                    print(f"Droplet {self.droplet_id} already destroyed.")
                    self.droplet_id = None
                    return
                resp.raise_for_status()
            except KeyboardInterrupt:
                # User pressed Ctrl-C during destroy — ignore and keep retrying.
                # The destroy MUST complete to stop billing.
                print(f"\n(Destroy in progress — please wait, this stops billing)")
            except requests.exceptions.RequestException as exc:
                if attempt < 3:
                    print(f"Destroy attempt {attempt} failed: {exc}. Retrying in 10s...")
                    try:
                        time.sleep(10)
                    except KeyboardInterrupt:
                        print(f"\n(Destroy in progress — please wait, this stops billing)")
                    continue
                raise

    def delete_all(self) -> None:
        """Alias for destroy(). Matches the executor.py cleanup call pattern."""
        self.destroy()

    def _droplet_ssh_keys(self) -> list[str | int]:
        if self.settings.ssh_keys:
            return list(self.settings.ssh_keys)
        if self.settings.register_ssh_key or self.settings.ssh_key_name or self.settings.ssh_public_key:
            return [self.ensure_ssh_key()]
        return []

    def _create_droplet_payload(
        self,
        droplet_name: str,
        region: str,
        size: str,
        ssh_keys: list[str | int],
    ) -> dict[str, Any]:
        return {
            "name": droplet_name,
            "region": region,
            "size": size,
            "image": self.settings.gpu_image,
            "ssh_keys": ssh_keys,
            "backups": False,
            "ipv6": False,
            "monitoring": False,
            "tags": list(self.settings.tags),
            "user_data": self._cloud_init_user_data(),
            "vpc_uuid": self.settings.vpc_uuid,
        }

    def _cloud_init_user_data(self) -> str:
        """Minimal cloud-init for native host execution.

        The AMD DevCloud Quick Start image already has PyTorch + ROCm
        pre-installed.  There is no Docker dependency — the runner executes
        workloads directly on the host via ``python3`` (see ``_run_host_stage``
        in ``docker_runner.py``).  Only ensure ``pip`` is available for the
        package-overlay install step that follows.
        """
        return """#cloud-config
package_update: true
packages:
  - python3-pip
runcmd:
  - [ bash, -lc, 'echo "Dynamic Cloud native host ready."' ]
"""

    def _request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        kwargs.setdefault("headers", self.headers)
        max_attempts = 4
        for attempt in range(1, max_attempts + 1):
            resp = getattr(requests, method.lower())(url, **kwargs)
            if resp.status_code != 429 or attempt == max_attempts:
                return resp
            retry_after = resp.headers.get("Retry-After") if hasattr(resp, "headers") else None
            try:
                delay = float(retry_after) if retry_after else min(2 ** attempt, 30)
            except ValueError:
                delay = min(2 ** attempt, 30)
            print(f"AMD DevCloud API rate limit hit; retrying in {delay:g}s ({attempt}/{max_attempts})...")
            time.sleep(delay)
        return resp

    # ── SSH Readiness ───────────────────────────────────────────────────

    def wait_for_ssh(self, ip: str, timeout_sec: int | None = None) -> None:
        """Poll until the configured user/key can run a command over SSH."""
        timeout = self.settings.wait_ssh_timeout_sec if timeout_sec is None else timeout_sec
        deadline = time.monotonic() + timeout
        last_error = ""
        while time.monotonic() < deadline:
            if not self._is_tcp_ssh_open(ip):
                last_error = "TCP port 22 is not accepting connections yet."
                time.sleep(5)
                continue

            result = subprocess.run(
                [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "StrictHostKeyChecking=accept-new",
                    "-o",
                    "UserKnownHostsFile=/dev/null",
                    "-o",
                    "ConnectTimeout=10",
                    "-i",
                    str(self.settings.ssh_private_key_path),
                    f"{self.settings.admin_user}@{ip}",
                    "true",
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode == 0:
                print("SSH is ready.")
                return
            last_error = (result.stderr or result.stdout or f"ssh exited with {result.returncode}").strip()
            time.sleep(5)
        raise TimeoutError(
            f"SSH login for {self.settings.admin_user}@{ip} was not ready within "
            f"{timeout} seconds. Last error: {last_error}"
        )

    @staticmethod
    def _is_tcp_ssh_open(ip: str) -> bool:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(3)
                sock.connect((ip, 22))
                return True
        except (ConnectionRefusedError, OSError):
            return False
