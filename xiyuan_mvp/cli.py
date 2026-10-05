from __future__ import annotations

import argparse
from pathlib import Path
import sys

from . import __version__
from .config import load_config
from .image_io import read_image
from .pipeline import StitchPipeline
from .run_io import save_failure, save_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="曦源两图无缝拼接")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("image_a")
    parser.add_argument("image_b")
    parser.add_argument("more_images", nargs="*", help="其他图片，按相邻重叠顺序输入")
    parser.add_argument("--output-dir", default="outputs/latest")
    parser.add_argument("--config")
    parser.add_argument("--ai", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--mask", help="同一配准画布下的灰度 Mask")
    args = parser.parse_args(argv)
    config = {}
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        print("输出目录已有文件，请指定一个新的目录，避免混淆实验。", file=sys.stderr)
        return 2
    try:
        config = load_config(args.config)
        if args.seed is not None:
            config["inpainting"]["seed"] = args.seed
        mask = read_image(args.mask)[:, :, 0] if args.mask else None
        pipe = StitchPipeline(config)
        progress = lambda message, percent: print(f"[{percent:3d}%] {message}", flush=True)
        if args.more_images:
            result = pipe.run_many([args.image_a,args.image_b,*args.more_images],use_ai=args.ai and mask is None,progress=progress)
            if mask is not None:
                result = pipe.run(args.image_a,args.image_b,prepared=result,use_ai=args.ai,custom_mask=mask,progress=progress)
        else:
            result = pipe.run(args.image_a, args.image_b, use_ai=args.ai, custom_mask=mask,progress=progress)
        save_run(output, result, config, extra={"image_a": args.image_a, "image_b": args.image_b})
        print(f"结果已写入：{output.resolve()}")
        return 0
    except Exception as exc:
        save_failure(output, exc, config)
        print(f"处理失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

