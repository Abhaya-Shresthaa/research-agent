from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(*_args, **_kwargs) -> bool:
        return False


ROOT_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT_DIR = ROOT_DIR


def _expand_path(value: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(value))).resolve()


def _validate_ssh_private_key(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"SSH private key not found at {path}. "
            "Set AMD_SSH_PRIVATE_KEY_PATH to the key that matches your Droplet SSH public key."
        )
    if os.name == "posix":
        mode = path.stat().st_mode & 0o777
        if mode & 0o077:
            raise PermissionError(
                f"SSH private key permissions are too open for {path}: {mode:04o}. "
                f"Run: chmod 600 {path}"
            )


def load_environment() -> None:
    protected_keys = set(os.environ)
    _load_env_file(ROOT_DIR / ".env", override=True, protected_keys=protected_keys)


def _load_env_file(path: Path, override: bool, protected_keys: set[str] | None = None) -> None:
    protected_values = {
        key: os.environ[key]
        for key in (protected_keys or set())
        if key in os.environ
    }
    loaded = load_dotenv(path, override=override)
    for key, value in protected_values.items():
        os.environ[key] = value
    if loaded or not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        if key in protected_values:
            continue
        if override or key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True)
class AmdSettings:
    """Configuration for AMD Developer Cloud Droplets."""
    api_key: str
    api_base_url: str = "https://api.devcloud.amd.com/v2"
    region: str = "atl1"
    vm_name: str = "dynamic-ml-droplet"
    admin_user: str = "root"
    default_size: str = "gpu-mi300x1-192gb-devcloud"
    gpu_image: str = "rocm-7-2-4"
    docker_base_image: str = ""
    docker_runtime_image: str = ""
    vpc_uuid: str = "8244dc95-5e6a-45b1-a667-d8c80a851d9b"
    project_id: str | None = None
    ssh_key_name: str | None = None
    ssh_public_key: str | None = None
    ssh_keys: tuple[str | int, ...] = ()
    register_ssh_key: bool = False
    tags: tuple[str, ...] = ()
    ssh_public_key_path: Path = field(default_factory=lambda: _expand_path("~/.ssh/id_rsa.pub"))
    ssh_private_key_path: Path = field(default_factory=lambda: _expand_path("~/.ssh/id_rsa"))
    wait_droplet_timeout_sec: int = 600
    wait_ssh_timeout_sec: int = 300
    wait_initialization_timeout_sec: int = 1200
    wait_provisioning_timeout_sec: int = 600  # _wait_for_provisioning in docker_runner
    regions: tuple[str, ...] = (
        "atl1",
    )
    vm_sizes: tuple[str, ...] = (
        "gpu-mi300x1-192gb-devcloud",
    )


@dataclass(frozen=True)
class AmdDropletSession:
    """A Droplet created before execution and reused by the training stage."""

    droplet_id: int
    ip: str
    vm_name: str
    remote_job_dir: str | None = None
    created_for_dataset_inspection: bool = False


@dataclass(frozen=True)
class LlmSettings:
    model: str = "accounts/fireworks/models/minimax-m2p7"
    api_key: str | None = None
    base_url: str | None = "https://api.fireworks.ai/inference/v1"


def load_amd_settings(vm_name: str | None = None, *, skip_ssh_validation: bool = False) -> AmdSettings:
    load_environment()
    api_key = _first_env_value("AMD_TOKEN", "AMD_API_KEY", "DIGITALOCEAN_ACCESS_TOKEN", "DO_API_TOKEN")
    if not api_key:
        raise RuntimeError(
            "AMD_TOKEN is required. Add AMD_TOKEN, AMD_API_KEY, or DIGITALOCEAN_ACCESS_TOKEN to "
            f"{ROOT_DIR / '.env'}, or export it in your shell."
        )

    default_size = (os.getenv("AMD_DEFAULT_VM_SIZE") or "gpu-mi300x1-192gb-devcloud").strip()
    if not default_size:
        raise RuntimeError("AMD_DEFAULT_VM_SIZE must not be empty.")

    raw_vm_sizes = os.getenv("AMD_VM_SIZES")
    if raw_vm_sizes:
        vm_sizes = tuple(size.strip() for size in raw_vm_sizes.split(",") if size.strip())
        if not vm_sizes:
            raise RuntimeError("AMD_VM_SIZES must contain at least one non-empty size.")
    else:
        vm_sizes = (default_size,)

    ssh_private_key_path = _expand_path(os.getenv("AMD_SSH_PRIVATE_KEY_PATH", "~/.ssh/id_rsa"))
    if not skip_ssh_validation:
        _validate_ssh_private_key(ssh_private_key_path)

    region = os.getenv("AMD_REGION", "atl1").strip() or "atl1"
    raw_regions = os.getenv("AMD_REGIONS")
    if raw_regions:
        regions = tuple(dict.fromkeys(item.strip() for item in raw_regions.split(",") if item.strip()))
        if not regions:
            raise RuntimeError("AMD_REGIONS must contain at least one non-empty region.")
    else:
        regions = tuple(dict.fromkeys((region, "atl1")))

    raw_ssh_keys = os.getenv("AMD_SSH_KEYS", "")
    ssh_keys = tuple(
        _coerce_ssh_key_value(item.strip())
        for item in raw_ssh_keys.split(",")
        if item.strip()
    )
    raw_tags = os.getenv("AMD_TAGS", "")
    tags = tuple(tag.strip() for tag in raw_tags.split(",") if tag.strip())
    register_ssh_key = (os.getenv("AMD_REGISTER_SSH_KEY", "").strip().lower() in {"1", "true", "yes", "on"})

    return AmdSettings(
        api_key=api_key,
        api_base_url=(os.getenv("AMD_API_BASE_URL") or "https://api.devcloud.amd.com/v2").strip().rstrip("/"),
        region=region,
        vm_name=vm_name or os.getenv("AMD_VM_NAME", "dynamic-ml-droplet"),
        admin_user=os.getenv("AMD_ADMIN_USER", "root"),
        default_size=default_size,
        gpu_image=(os.getenv("AMD_GPU_IMAGE") or "rocm-7-2-4").strip(),
        docker_base_image=(os.getenv("AMD_DOCKER_BASE_IMAGE") or "").strip(),
        docker_runtime_image=(os.getenv("AMD_DOCKER_RUNTIME_IMAGE") or "").strip(),
        vpc_uuid=(os.getenv("AMD_VPC_UUID") or "8244dc95-5e6a-45b1-a667-d8c80a851d9b").strip(),
        project_id=os.getenv("AMD_PROJECT_ID") or None,
        ssh_key_name=os.getenv("AMD_SSH_KEY_NAME") or None,
        ssh_public_key=(os.getenv("AMD_SSH_PUBLIC_KEY") or "").strip() or None,
        ssh_keys=ssh_keys,
        register_ssh_key=register_ssh_key,
        tags=tags,
        ssh_public_key_path=_expand_path(os.getenv("AMD_SSH_PUBLIC_KEY_PATH", "~/.ssh/id_rsa.pub")),
        ssh_private_key_path=ssh_private_key_path,
        wait_droplet_timeout_sec=int(os.getenv("AMD_DROPLET_WAIT_TIMEOUT_SEC", "600")),
        wait_ssh_timeout_sec=int(os.getenv("AMD_SSH_WAIT_TIMEOUT_SEC", "600")),
        wait_initialization_timeout_sec=int(os.getenv("AMD_INITIALIZATION_WAIT_TIMEOUT_SEC", "1200")),
        wait_provisioning_timeout_sec=int(os.getenv("AMD_PROVISIONING_WAIT_TIMEOUT_SEC", "600")),
        regions=regions,
        vm_sizes=vm_sizes,
    )


def _first_env_value(*keys: str) -> str | None:
    for key in keys:
        value = (os.getenv(key) or "").strip()
        if value:
            return value
    return None


def _coerce_ssh_key_value(value: str) -> str | int:
    return int(value) if value.isdigit() else value


def load_llm_settings() -> LlmSettings:
    load_environment()
    return LlmSettings(
        model=os.getenv("MODEL1") or os.getenv("MODEL1_MODEL") or "accounts/fireworks/models/minimax-m2p7",
        api_key=os.getenv("FIREWORKS_API_KEY") or None,
        base_url=os.getenv("FIREWORKS_BASE_URL") or "https://api.fireworks.ai/inference/v1",
    )
