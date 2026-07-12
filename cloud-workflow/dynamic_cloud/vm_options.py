from __future__ import annotations

from typing import Any


VM_OPTIONS: list[dict[str, Any]] = [
    {
        "size": "gpu-mi300x1-192gb-devcloud",
        "label": "MI300X x1 — Single GPU (default)",
        "hardware": "1 GPU - 192 GB VRAM - 20 vCPU - 240 GB RAM",
        "storage": "Boot: 720 GB NVMe, Scratch: 5 TB NVMe",
        "cost_note": "$1.99/GPU/hr. Destroyed after run to stop billing.",
        "supports_gpu": True,
    },
    {
        "size": "gpu-mi300x8-1536gb-devcloud",
        "label": "MI300X x8 — Multi-GPU",
        "hardware": "8 GPU - 1.5 TB VRAM - 160 vCPU - 1920 GB RAM",
        "storage": "Boot: 2 TB NVMe, Scratch: 40 TB NVMe",
        "cost_note": "$15.92/hr (8×$1.99/GPU/hr). Destroyed after run to stop billing.",
        "supports_gpu": True,
    },
]


AMD_GPU_PRICING_PER_HOUR: dict[str, float] = {
    "gpu-mi300x1-192gb-devcloud": 1.99,
    "gpu-mi300x8-1536gb-devcloud": 15.92,
}


# Available Quick Start images on AMD Developer Cloud
AMD_GPU_IMAGES: dict[str, dict[str, str]] = {
    "rocm": {
        "slug": "rocm-7-2-4",
        "label": "ROCm Software (base)",
        "description": "ROCm 7.2.4 on Ubuntu 24.04. Install any AI tools you need.",
    },
    "pytorch": {
        "slug": "amddevelopercloud-pytorch2100rocm724",
        "label": "PyTorch Quick Start",
        "description": "PyTorch 2.10.0 on ROCm 7.2.4, Ubuntu 24.04.",
    },
    "tensorflow": {
        "slug": "rocm-7-2-4",
        "label": "ROCm Software (base)",
        "description": "ROCm 7.2.4 on Ubuntu 24.04. Use the TensorFlow ROCm Docker runtime image inside this Droplet.",
    },
    "vllm": {
        "slug": "amddeveloperclou-vllm0230rocm724",
        "label": "vLLM Quick Start",
        "description": "vLLM 0.23.0 on ROCm 7.2.4, Ubuntu 24.04.",
    },
    "sglang": {
        "slug": "amddeveloperclou-sglang0514rocm72",
        "label": "SGLang Quick Start",
        "description": "SGLang 0.5.14 on ROCm 7.2.4, Ubuntu 24.04.",
    },
    "jax": {
        "slug": "amddeveloperclou-jax082rocm724",
        "label": "JAX Quick Start",
        "description": "JAX 0.8.2 on ROCm 7.2.4, Ubuntu 24.04.",
    },
}


def vm_option_for_size(vm_size: str, saved_job_spec: dict[str, Any] | None = None) -> dict[str, Any]:
    """Look up a VM option by size slug."""
    for option in VM_OPTIONS:
        if option["size"] == vm_size:
            return dict(option)

    return {
        "size": vm_size,
        "label": "Custom GPU Droplet",
        "hardware": "Custom AMD GPU plan",
        "cost_note": "Custom plan — check DigitalOcean pricing.",
        "supports_gpu": True,
    }


# ── Image Selection Menu ────────────────────────────────────────────
#
# Matching the pattern from ano_temp/manage_gpu.py — some image slugs
# differ between 1× MI300X and 8× MI300X topologies.

AMD_GPU_IMAGE_OPTIONS: list[dict[str, str]] = [
    {
        "name": "PyTorch (v2.10.0 — Deep Learning Framework) ✅ Recommended",
        "slug_1x": "amddevelopercloud-pytorch2100rocm724",
        "slug_8x": "amddevelopercloud-pytorch2100rocm724",
        "name_prefix": "2.10.0-",
    },
    {
        "name": "ROCm™ Software (Base ROCm 7.2.4 Environment)",
        "slug_1x": "rocm-7-2-4",
        "slug_8x": "rocm-7-2-4",
        "name_prefix": "rocm-7-2-4-",
    },
    {
        "name": "vLLM (v0.23.0 — High-throughput LLM serving)",
        "slug_1x": "amddeveloperclou-vllm0230rocm724",
        "slug_8x": "amddeveloperclou-vllm0230rocm724",
        "name_prefix": "0.23.0-",
    },
    {
        "name": "SGLang (v0.5.14 — Fast LLM serving runtime)",
        "slug_1x": "amddeveloperclou-sglang0514rocm72",
        "slug_8x": "sglang-0-5-14-rocm-7-2-4",
        "name_prefix": "0.5.14-",
    },
    {
        "name": "JAX (v0.8.2 — Numerical Computing & ML Framework)",
        "slug_1x": "amddeveloperclou-jax082rocm724",
        "slug_8x": "jax-0-8-2-rocm-7-2-4",
        "name_prefix": "0.8.2-",
    },
    {
        "name": "Ubuntu Base OS (v26.04 x64) ⚠️ Requires Docker install",
        "slug_1x": "ubuntu-26-04-x64",
        "slug_8x": "ubuntu-26-04-x64",
        "name_prefix": "ubuntu-",
    },
]


def _vm_size_suffix(vm_size: str) -> str:
    """Convert a VM size slug to the AMD Cloud node-name suffix.

    e.g. ``gpu-mi300x1-192gb-devcloud`` → ``mi300x1-192gb-devcloud-atl1``
    """
    return vm_size.replace("gpu-", "") + "-atl1"


def select_amd_image(vm_size: str) -> tuple[str, str]:
    """Show an interactive menu for selecting the AMD GPU Droplet image.

    Returns ``(image_slug, node_name)`` where *node_name* is a descriptive
    name that includes the image version prefix and VM size suffix so it
    is easy to identify in the AMD Cloud console.

    Slugs are automatically resolved for the right topology (1× vs 8×).
    """
    is_8x = "mi300x8" in vm_size
    suffix = _vm_size_suffix(vm_size)

    other_options = AMD_GPU_IMAGE_OPTIONS[1:]  # all except PyTorch

    print("\n" + "=" * 56)
    print("  Select an Image to Deploy on the GPU Droplet")
    print("=" * 56)
    print(f"  [1] {AMD_GPU_IMAGE_OPTIONS[0]['name']}")
    print("  [2] Others")

    while True:
        try:
            choice = input("\nEnter image choice number (1-2): ").strip()
            if not choice or choice == "1":
                selected = AMD_GPU_IMAGE_OPTIONS[0]
                slug = selected["slug_8x"] if is_8x else selected["slug_1x"]
                node_name = f"{selected['name_prefix']}{suffix}"
                print(f"  Chosen: {selected['name']}")
                print(f"  Image slug: {slug}")
                print(f"  Droplet name: {node_name}")
                print("=" * 56)
                return slug, node_name

            if choice == "2":
                print()
                for idx, opt in enumerate(other_options, 1):
                    print(f"  [{idx}] {opt['name']}")
                print(f"  [{len(other_options) + 1}] PyTorch (back to main)")

                sub_choice = input(
                    f"\nEnter image choice number (1-{len(other_options) + 1}): "
                ).strip()
                if not sub_choice:
                    continue
                sub_idx = int(sub_choice) - 1
                if 0 <= sub_idx < len(other_options):
                    selected = other_options[sub_idx]
                    slug = selected["slug_8x"] if is_8x else selected["slug_1x"]
                    node_name = f"{selected['name_prefix']}{suffix}"
                    print(f"  Chosen: {selected['name']}")
                    print(f"  Image slug: {slug}")
                    print(f"  Droplet name: {node_name}")
                    print("=" * 56)
                    return slug, node_name
                if sub_idx == len(other_options):
                    continue  # back to main menu
                print(f"  Please choose a number between 1 and {len(other_options) + 1}.")
                continue

            print("  Please choose 1 or 2.")
        except ValueError:
            print("  Input must be a valid option number.")


def validate_vm_accelerator(selected_vm: dict[str, Any], accelerator: str) -> None:
    """Validate that the selected plan supports the requested accelerator."""
    if str(accelerator).strip().lower() == "gpu" and not selected_vm.get("supports_gpu"):
        raise ValueError(
            f"Selected plan {selected_vm.get('size')} does not provide a GPU. "
            "Choose an MI300X plan."
        )
