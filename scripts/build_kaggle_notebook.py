"""Build a self-contained private notebook from an explicit source allowlist.

Run from the repository root with python -m scripts.build_kaggle_notebook --owner USER.
Only Python source and default configuration are embedded; no local data or credentials.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "xiyuan_mvp/__init__.py", "xiyuan_mvp/types.py", "xiyuan_mvp/errors.py",
    "xiyuan_mvp/config.py", "xiyuan_mvp/image_io.py", "xiyuan_mvp/registration.py",
    "xiyuan_mvp/seam_mask.py", "xiyuan_mvp/blending.py", "xiyuan_mvp/inpainting.py",
    "xiyuan_mvp/refinement.py", "xiyuan_mvp/loftr.py",
    "xiyuan_mvp/neural_alignment.py", "xiyuan_mvp/structural_repair.py", "xiyuan_mvp/tiled_inpainting.py",
    "xiyuan_mvp/ai_guard.py", "xiyuan_mvp/clear_fusion.py", "xiyuan_mvp/pipeline.py", "xiyuan_mvp/cli.py", "xiyuan_mvp/run_io.py", "xiyuan_mvp/default.yaml", "configs/default.yaml",
    "scripts/generate_demo_pair.py", "scripts/kaggle_smoke.py",
)


def cell(kind: str, source: str, number: int) -> dict:
    result = {"id": f"xiyuan-{number}", "cell_type": kind, "metadata": {},
              "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def build_notebook() -> tuple[dict, dict]:
    sources = {path: (ROOT / path).read_text(encoding="utf-8") for path in SOURCE_FILES}
    manifest = {path: hashlib.sha256(text.encode()).hexdigest() for path, text in sources.items()}
    bootstrap = (
        "from pathlib import Path\nimport sys\n"
        "ROOT = Path('/kaggle/working/xiyuan_source')\nROOT.mkdir(exist_ok=True)\n"
        f"SOURCES = {sources!r}\n"
        "for relative, text in SOURCES.items():\n"
        "    target = ROOT / relative\n    target.parent.mkdir(parents=True, exist_ok=True)\n"
        "    target.write_text(text, encoding='utf-8')\n"
        "sys.path.insert(0, str(ROOT))\n"
        f"print('Source bundle SHA256:', {hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()!r})\n"
    )
    cells = [
        cell("markdown", "# 曦源 MVP 云端 GPU 验证\n\n"
             "运行项目真实的 SD Inpainting 分支，使用程序生成的测试图；不包含本地照片或凭据。\n"
             "免费 GPU Notebook，非长期在线服务。批量运行上限由 push 的 --timeout 1800 设置。\n"
             "先完成下面的 GPU 检查，再安装依赖。结果保存在 /kaggle/working/results。\n", 0),
        cell("code", "import json\nfrom pathlib import Path\nimport torch\n"
             "assert torch.cuda.is_available(), 'CUDA unavailable: select a free GPU accelerator in Session options.'\n"
             "gpu = {'name': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'cuda': torch.version.cuda,\n"
             "       'vram_gb': round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)}\n"
             "print(json.dumps(gpu, indent=2))\n"
             "Path('/kaggle/working/gpu.json').write_text(json.dumps(gpu, indent=2))\n"
             "# A CUDA presence check alone does not detect unsupported GPU architectures.\n"
             "assert (torch.ones(4, device='cuda') * 2).sum().item() == 8\n"
             "torch.cuda.synchronize()\nprint('CUDA arithmetic check passed')\n", 1),
        cell("code", "import subprocess, sys\n"
             "subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet',\n"
             "    'diffusers==0.35.1', 'transformers==4.56.2', 'accelerate==1.10.1',\n"
             "    'safetensors>=0.4,<1', 'Pillow>=10', 'PyYAML>=6'], check=True, timeout=600)\n"
             "# Keep Kaggle's existing PyTorch/CUDA build. No local GUI dependencies are required.\n", 2),
        cell("code", bootstrap, 3),
        cell("markdown", "## 实验配置\n\n默认 512×512、20 步、两个不同强度的合成纹理测试。"
             "权重固定到运行时查询到的仓库 revision，并写入报告。此测试检查真实推理链路，不代表真实照片质量已达标。\n", 4),
        cell("code", "import runpy\nrunpy.run_path(str(ROOT / 'scripts/kaggle_smoke.py'), run_name='__main__')\n", 5),
        cell("code", "from IPython.display import display, Image\n"
             "for path in sorted(Path('/kaggle/working/results').glob('*/comparison.jpg')):\n"
             "    print(path.parent.name)\n    display(Image(filename=str(path)))\n"
             "print(Path('/kaggle/working/results/report.json').read_text())\n", 6),
    ]
    notebook = {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }, "nbformat": 4, "nbformat_minor": 5}
    return notebook, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner", required=True, help="Kaggle username, never an API key")
    parser.add_argument("--output", type=Path, default=ROOT / "kaggle")
    args = parser.parse_args()
    if not args.owner.replace("_", "").isalnum():
        parser.error("Invalid Kaggle username")
    notebook, manifest = build_notebook()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "xiyuan_gpu.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "source-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    metadata = {"id": f"{args.owner}/xiyuan-stitch-gpu-mvp", "title": "Xiyuan Stitch GPU MVP",
                "code_file": "xiyuan_gpu.ipynb", "language": "python", "kernel_type": "notebook",
                "is_private": True, "enable_gpu": True, "enable_tpu": False,
                "machine_shape": "NvidiaTeslaT4",
                "enable_internet": True, "dataset_sources": [], "competition_sources": [],
                "kernel_sources": [], "model_sources": []}
    (args.output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Prepared private notebook: {metadata['id']}; {len(manifest)} allowlisted source files")


if __name__ == "__main__":
    main()
