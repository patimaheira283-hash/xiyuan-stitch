"""Run real, bounded GPU inference and preserve results for local download."""
from __future__ import annotations

import importlib.metadata
import json
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
from huggingface_hub import model_info

from xiyuan_mvp.config import load_config
from xiyuan_mvp.image_io import write_image
from xiyuan_mvp.pipeline import StitchPipeline


def brick_pair() -> tuple[np.ndarray, np.ndarray]:
    """Procedural flat wall, with shared features and slight local misalignment."""
    rng = np.random.default_rng(9026)
    wall = np.full((512, 1056, 3), (160, 173, 180), dtype=np.uint8)
    for row in range(9):
        for col in range(-1, 10):
            x0 = col * 136 + (68 if row % 2 else 0)
            y0 = row * 64
            cv2.rectangle(wall, (x0 + 5, y0 + 5), (x0 + 131, y0 + 59),
                          tuple(int(v) for v in rng.integers([55, 72, 132], [95, 122, 205])), -1)
    noise = rng.normal(0, 5, wall.shape)
    wall = np.clip(wall.astype(float) + noise, 0, 255).astype(np.uint8)
    for _ in range(160):
        x, y = int(rng.integers(10, 1040)), int(rng.integers(10, 500))
        cv2.circle(wall, (x, y), int(rng.integers(1, 4)), (75, 95, 133), -1)
    first, second = wall[:, :720].copy(), wall[:, 336:].copy()
    ys, xs = np.indices(second.shape[:2], dtype=np.float32)
    bump = 3.0 * np.exp(-((xs - 190) / 80) ** 2) * np.sin(ys / 60)
    second = cv2.remap(second, xs, ys + bump, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    second = cv2.convertScaleAbs(second, alpha=1.02, beta=3)
    return first, second


def main() -> None:
    output = Path('/kaggle/working/results')
    output.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "test_type": "synthetic GPU integration smoke", "cases": []}
    started = time.perf_counter()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required. This test never falls back to CPU.")
        config = load_config()
        config['inpainting'].update(device='cuda', steps=20, variant='fp16', local_files_only=False,
                                    cache_dir='/kaggle/temp/xiyuan-hf-cache',
                                    prompt='a red brick wall, continuous mortar joints, consistent texture and lighting')
        revision = model_info(config['inpainting']['model_id']).sha
        if not revision:
            raise RuntimeError("Could not resolve model revision")
        config['inpainting']['revision'] = revision
        report.update(model_id=config['inpainting']['model_id'], model_revision=revision,
                      model_variant=config['inpainting']['variant'],
                      gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                      cuda=torch.version.cuda, seed=config['inpainting']['seed'],
                      packages={name: importlib.metadata.version(name) for name in
                                ('diffusers', 'transformers', 'accelerate', 'numpy', 'Pillow')})
        print('MODEL', report['model_id'], revision, flush=True)
        print('GPU', report['gpu'], flush=True)
        first, second = brick_pair()
        pipeline = StitchPipeline(config)
        for strength in (0.35, 0.50):
            name = f'brick-strength-{strength:.2f}'
            print('CASE', name, flush=True)
            config['inpainting']['strength'] = strength
            torch.cuda.reset_peak_memory_stats()
            result = pipeline.run(first, second, use_ai=True,
                                  progress=lambda message, n: print(n, message, flush=True))
            torch.cuda.synchronize()
            folder = output / name
            write_image(folder / 'input_a.png', first)
            write_image(folder / 'input_b.png', second)
            write_image(folder / 'traditional.png', result.traditional_image)
            write_image(folder / 'mask.png', result.mask.binary_mask)
            write_image(folder / 'soft_mask.png', result.mask.soft_mask)
            write_image(folder / 'ai_result.png', result.final_image)
            comparison = np.concatenate([result.traditional_image, result.final_image], axis=0)
            cv2.putText(comparison, 'Traditional (top) / SD Inpainting (bottom)', (12, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2)
            write_image(folder / 'comparison.jpg', comparison)
            difference = np.abs(result.final_image.astype(np.int16) - result.traditional_image.astype(np.int16))
            outside = result.mask.soft_mask == 0
            if np.any(difference[outside]):
                raise AssertionError('Pixels outside the soft mask were modified')
            if not np.any(difference):
                raise AssertionError('Inference returned an unchanged canvas')
            case = {"name": name, "strength": strength, **result.metrics,
                    "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3),
                    "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 3),
                    "outside_mask_max_difference": int(difference[outside].max()) if np.any(outside) else 0,
                    "inside_mask_mean_difference": float(difference[result.mask.binary_mask > 0].mean())}
            report['cases'].append(case)
            (folder / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
            print(json.dumps(case), flush=True)
        report['status'] = 'passed'
    except Exception as exc:
        report.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        traceback.print_exc()
        raise
    finally:
        report['elapsed_seconds'] = time.perf_counter() - started
        (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print('REPORT', json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
