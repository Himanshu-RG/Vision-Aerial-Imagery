"""Record the software/hardware environment of this machine as JSON.

Usage:
    python scripts/env_audit.py                 # writes outputs/env/<hostname>.json
    python scripts/env_audit.py --out FILE.json
Run it on every machine that produces results; the JSON is committed with the code.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

PACKAGES = [
    "torch", "torchvision", "transformers", "huggingface_hub", "safetensors", "open_clip_torch",
    "pycocotools", "numpy", "pillow", "pyyaml", "pydantic", "pandas", "matplotlib", "psutil", "pytest",
]


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=20, cwd=REPO_ROOT)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def package_versions() -> dict[str, str | None]:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            versions[name] = None
    return versions


def torch_info() -> dict:
    try:
        import torch
    except ImportError:
        return {"installed": False}
    info = {
        "installed": True,
        "version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "devices": [],
    }
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            info["devices"].append({
                "index": i,
                "name": props.name,
                "capability": f"{props.major}.{props.minor}",
                "total_mem_gb": round(total / 1024**3, 2),
                "free_mem_gb_at_audit": round(free / 1024**3, 2),
            })
    return info


def nvidia_smi() -> list[dict] | None:
    out = _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used",
                "--format=csv,noheader,nounits"])
    if not out:
        return None
    gpus = []
    for line in out.splitlines():
        name, driver, total, used = [x.strip() for x in line.split(",")]
        gpus.append({"name": name, "driver": driver, "mem_total_mb": int(total), "mem_used_mb": int(used)})
    return gpus


def disk_info() -> dict[str, float]:
    paths = {"repo": REPO_ROOT}
    store = os.environ.get("AOVD_STORE")
    if not store:
        local = REPO_ROOT / "configs" / "paths.local.yaml"
        if local.is_file():
            import yaml

            store = (yaml.safe_load(local.read_text(encoding="utf-8")) or {}).get("store")
    if store and Path(store).exists():
        paths["store"] = Path(store)
    return {k: round(shutil.disk_usage(p).free / 1024**3, 1) for k, p in paths.items()}


def collect() -> dict:
    try:
        import psutil

        ram_gb = round(psutil.virtual_memory().total / 1024**3, 1)
    except ImportError:
        ram_gb = None
    return {
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "machine": platform.machine(),
        "cpu": platform.processor(),
        "cpu_count": os.cpu_count(),
        "ram_gb": ram_gb,
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "packages": package_versions(),
        "torch": torch_info(),
        "nvidia_smi": nvidia_smi(),
        "disk_free_gb": disk_info(),
        "git_commit": _run(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(_run(["git", "status", "--porcelain"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    info = collect()
    out = args.out or REPO_ROOT / "outputs" / "env" / f"{info['hostname']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
