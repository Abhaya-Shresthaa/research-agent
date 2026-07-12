from __future__ import annotations

import json
import urllib.request
from typing import Any


def resolve_hf_dataset_id(raw_id: str) -> str:
    """Resolve a HuggingFace dataset ID to its canonical namespace/name form.

    For example, ``"beans"`` -> ``"AI-Lab-Makerere/beans"``.

    Uses the HuggingFace Hub API to discover the namespace.  If the API call
    fails (offline, timeout, or the dataset simply doesn't exist) the original
    ID is returned as-is so the upstream caller can still fail gracefully.
    Results are cached via a function attribute so that repeated resolution
    of the same ID reuses the API call.
    """
    _cache = vars(resolve_hf_dataset_id).setdefault("_cache", {})
    if raw_id in _cache:
        canonical = _cache[raw_id]
        return canonical if canonical is not None else raw_id

    if "/" in raw_id:
        _cache[raw_id] = raw_id
        return raw_id

    try:
        url = f"https://huggingface.co/api/datasets/{urllib.request.quote(raw_id, safe='')}"
        req = urllib.request.Request(url, headers={"User-Agent": "dynamic-cloud-workflow/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        canonical_id = data.get("id", raw_id)
        _cache[raw_id] = canonical_id
        if canonical_id != raw_id:
            print(
                f"[dataset] Resolved HuggingFace dataset '{raw_id}' to "
                f"'{canonical_id}' (namespace auto-detected).",
                flush=True,
            )
        return canonical_id
    except Exception as exc:
        _cache[raw_id] = None
        print(
            f"[dataset] Could not auto-resolve HuggingFace dataset ID "
            f"'{raw_id}' (API error: {exc}). Trying '{raw_id}' as-is.",
            flush=True,
        )
        return raw_id


def normalize_dataset_config(
    dataset: dict[str, Any] | None,
    local_container_data_dir: str = "/workspace/data",
    remote_container_data_dir: str = "/workspace/prepared_datasets",
) -> dict[str, Any]:
    config = dict(dataset or {})
    raw_type = str(config.get("type") or "").strip().lower()
    aliases = {
        "hf": "huggingface",
        "hugging-face": "huggingface",
        "hugging_face": "huggingface",
        "git": "github",
    }
    dataset_type = aliases.get(raw_type, raw_type)
    url = str(config.get("url") or config.get("uri") or "").strip()
    identifier = str(config.get("id") or config.get("repo") or "").strip()

    supported_types = {"", "none", "local", "huggingface", "github"}
    if raw_type and dataset_type not in supported_types:
        raise ValueError(
            "Unsupported dataset.type: "
            f"{raw_type}. Supported dataset sources are local, huggingface, and github."
        )

    if not dataset_type:
        if config.get("local_paths") or config.get("path"):
            dataset_type = "local"
        elif "github.com" in url:
            dataset_type = "github"
        elif identifier:
            dataset_type = "huggingface"
        else:
            dataset_type = "none"

    if config.get("path") and not config.get("local_paths"):
        config["local_paths"] = [config["path"]]

    if dataset_type == "huggingface":
        raw_id = identifier or str(config.get("name") or "").strip()
        revision = config.get("revision")
        if revision is None and "@" in raw_id:
            raw_id, revision = raw_id.split("@", 1)
            config["revision"] = revision
        # Auto-resolve bare names (e.g. "beans") to canonical namespace/name form
        # (e.g. "AI-Lab-Makerere/beans").  This avoids HfUriError on newer
        # huggingface_hub that requires namespace/name format.
        if "/" not in raw_id:
            resolved = resolve_hf_dataset_id(raw_id)
            if "/" in resolved:
                raw_id = resolved
        config["id"] = raw_id
        config.setdefault("repo", config["id"])
    elif dataset_type == "github" and url:
        config["url"] = url

    config["type"] = dataset_type
    if dataset_type == "local":
        config.setdefault("container_data_dir", local_container_data_dir)
    elif dataset_type != "none":
        config.setdefault("container_data_dir", remote_container_data_dir)
    return config
