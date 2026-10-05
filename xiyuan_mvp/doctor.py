from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .run_io import environment, write_json


def main():
    parser = argparse.ArgumentParser(description="检查桌面和 AI 运行环境")
    parser.add_argument("--output")
    args = parser.parse_args()
    report = environment()
    report["config_readable"] = bool(load_config())
    try:
        import torch
        report["cuda_available"] = torch.cuda.is_available()
        report["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        report["total_vram_mb"] = torch.cuda.get_device_properties(0).total_memory/1024**2 if torch.cuda.is_available() else None
    except ImportError:
        report["cuda_available"] = False
        report["gpu"] = None
    report["ai_note"] = "CUDA available; model loading still needs verification." if report["cuda_available"] else "No CUDA. Use the free GPU notebook; desktop baselines and mask editing remain available."
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
