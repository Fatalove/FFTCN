"""里程碑 1 环境诊断：不依赖项目模块，可在任意候选环境中运行。"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def package_version(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
    except (ImportError, OSError):
        return None
    return str(getattr(module, "__version__", "unknown"))


def gpu_driver_info() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        return {"available": False, "error": str(exc)}

    devices = []
    for line in result.stdout.splitlines():
        name, driver, memory_mib = (part.strip() for part in line.split(",", 2))
        devices.append(
            {"name": name, "driver": driver, "memory_mib": int(memory_mib)}
        )
    return {"available": bool(devices), "devices": devices}


def torch_info() -> dict[str, Any]:
    try:
        import torch
    except (ImportError, OSError) as exc:
        return {"installed": False, "error": str(exc)}

    cuda_available = torch.cuda.is_available()
    return {
        "installed": True,
        "version": torch.__version__,
        "compiled_cuda_runtime": torch.version.cuda,
        "cuda_available": cuda_available,
        "device_count": torch.cuda.device_count(),
        "devices": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
        if cuda_available
        else [],
    }


def collect_report() -> dict[str, Any]:
    return {
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "prefix": sys.prefix,
            "platform": platform.platform(),
        },
        "packages": {
            name: package_version(name)
            for name in (
                "numpy", "scipy", "matplotlib", "tqdm", "pyedflib", "openpyxl"
            )
        },
        "nvidia_driver": gpu_driver_info(),
        "torch": torch_info(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        help="可选：将同一份诊断结果保存为 UTF-8 JSON。",
    )
    args = parser.parse_args()

    report = collect_report()
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
