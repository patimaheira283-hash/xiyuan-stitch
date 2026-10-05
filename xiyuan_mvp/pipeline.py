from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time
from typing import Callable

import cv2
import numpy as np

from .blending import blend_generated_patch, feather_blend, poisson_blend
from .ai_guard import apply_ai_guard
from .config import load_config, validate_config
from .errors import CancelledError, StitchError
from .image_io import ensure_bgr, read_image
from .inpainting import DiffusersInpainter, extract_patch, restore_patch
from .registration import register_images
from .refinement import prepare_structural_base
from .seam_mask import generate_seam_mask, mask_from_binary
from .types import MaskResult, StitchResult

ProgressCallback = Callable[[str, int], None]


class StitchPipeline:
    def __init__(self, config: dict | None = None):
        self.config = config if config is not None else load_config()
        self._inpainter: DiffusersInpainter | None = None

    @staticmethod
    def _notify(callback: ProgressCallback | None, message: str, percent: int) -> None:
        if callback:
            callback(message, percent)

    def run(self, image_a: str | Path | np.ndarray, image_b: str | Path | np.ndarray, *,
            use_ai: bool = False, custom_mask: np.ndarray | None = None,
            progress: ProgressCallback | None = None, prepared: StitchResult | None = None,
            cancelled: Callable[[], bool] | None = None,
            valid_mask_a: np.ndarray | None = None) -> StitchResult:
        started = time.perf_counter()
        validate_config(self.config)
        if cancelled and cancelled():
            raise CancelledError("已取消。")
        timings = {}
        if prepared is None:
            self._notify(progress, "读取图片", 5)
            first = ensure_bgr(read_image(image_a) if isinstance(image_a, (str, Path)) else image_a)
            second = ensure_bgr(read_image(image_b) if isinstance(image_b, (str, Path)) else image_b)
            self._notify(progress, "自动配准", 20)
            t = time.perf_counter()
            registration = register_images(first, second, self.config["registration"], valid_mask_a=valid_mask_a)
            timings["registration_seconds"] = time.perf_counter() - t
            self._notify(progress, "生成羽化与泊松融合基线", 45)
            t = time.perf_counter()
            traditional = feather_blend(registration)
            timings["feather_seconds"] = time.perf_counter() - t
            t = time.perf_counter()
            poisson = poisson_blend(registration)
            timings["poisson_seconds"] = time.perf_counter() - t
        else:
            registration = prepared.registration
            traditional = prepared.traditional_image
            poisson = prepared.poisson_image
            timings = {k: v for k, v in prepared.metrics.items() if k in (
                "registration_seconds", "feather_seconds", "poisson_seconds")}
        self._notify(progress, "生成接缝区域", 60)
        refinement_metrics={}
        mask_registration=registration
        if prepared is not None:
            repair_base=prepared.repair_base_image if prepared.repair_base_image is not None else traditional
            refinement_metrics={k:v for k,v in prepared.metrics.items() if k.startswith(("refinement_","exposure_","overlap_mae_","flow_","local_flow_","clear_"))}
        elif self.config.get('clear_fusion',{}).get('enabled',False):
            self._notify(progress,'比较原图清晰度并融合纹理',55)
            from .clear_fusion import fuse
            repair_base,clear_metrics=fuse(registration,self.config['clear_fusion'])
            refinement_metrics={**{'clear_'+k:v for k,v in clear_metrics.items()},'clear_fusion_enabled':True}
        elif self.config.get("refinement",{}).get("enabled",False):
            repair_base,mask_registration,refinement_metrics=prepare_structural_base(registration,self.config["refinement"])
        else:
            repair_base=traditional
        allowed = prepared.repair_region if prepared is not None and prepared.repair_region is not None else registration.overlap_mask
        mask_result = prepared.mask if prepared is not None else generate_seam_mask(mask_registration, self.config["mask"])
        if custom_mask is not None:
            if custom_mask.shape != traditional.shape[:2]:
                raise StitchError("Mask 尺寸与当前画布不同，请重新生成接缝区域。")
            custom = np.where(custom_mask > 127, 255, 0).astype(np.uint8)
            custom = cv2.bitwise_and(custom, allowed)
            ys, xs = np.where(custom > 0)
            if not xs.size:
                raise StitchError("手动 Mask 与两图重叠区域没有交集。")
            radius = int(self.config["mask"]["feather_radius"])
            soft = cv2.GaussianBlur(custom, (radius * 2 + 1, radius * 2 + 1), 0)
            soft = cv2.bitwise_and(soft, allowed)
            bbox = (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1))
            mask_result = MaskResult(custom, soft, bbox, "custom")
        final = repair_base.copy()
        ai_seconds = 0.0
        ai_metrics = {}
        ai_overall_started = time.perf_counter()
        engine = self.config['inpainting'].get('engine','diffusion')
        if use_ai and engine in ('neural_alignment','hybrid'):
            if prepared is not None and prepared.metrics.get('sequence_count',2)>2:
                raise StitchError('本机结构修复暂支持两图。多图接缝请使用生成修复或分对处理。')
            self._notify(progress, '神经网络估计局部错位', 70)
            from .neural_alignment import repair_alignment
            ai_started = time.perf_counter()
            neural_config = deepcopy(self.config['neural_alignment'])
            neural_config['clear_fusion'] = deepcopy(self.config['clear_fusion'])
            final, ai_metrics = repair_alignment(registration, repair_base, mask_result,
                neural_config, method=neural_config['method'])
            ai_seconds = time.perf_counter() - ai_started
            if engine == 'hybrid':
                repair_base = final.copy()
                ai_metrics['neural_repair_seconds'] = ai_seconds
        diffusion_allowed = use_ai and engine in ('diffusion', 'hybrid')
        diffusion_skip_reason = None
        if diffusion_allowed and engine == 'hybrid' and self.config['inpainting'].get('hybrid_diffusion_policy', 'always') == 'on_structural_gain':
            residual = ai_metrics.get('structural_error_after')
            if not ai_metrics.get('repair_applied', False):
                diffusion_allowed = False
                diffusion_skip_reason = 'no_confirmed_structural_gain'
            elif residual is not None and float(residual) <= float(self.config['inpainting'].get('hybrid_min_residual_error', 2.0)):
                diffusion_allowed = False
                diffusion_skip_reason = 'residual_below_generation_threshold'
        if diffusion_allowed:
            ai_started = time.perf_counter()
            ai_config = self.config["inpainting"]
            if self._inpainter is None:
                self._inpainter = DiffusersInpainter(deepcopy(ai_config))
            if isinstance(self._inpainter, DiffusersInpainter):
                self._inpainter.config = deepcopy(ai_config)
            def generate(patch, patch_mask, callback):
                if isinstance(self._inpainter, DiffusersInpainter):
                    image = self._inpainter.generate(patch, patch_mask, progress=callback, cancelled=cancelled)
                    return image, deepcopy(self._inpainter.last_metrics)
                return self._inpainter.generate(patch, patch_mask), {"test_double": True}
            if ai_config.get("patch_layout", "single") == "native_tiled":
                from .tiled_inpainting import repair_tiled
                final, generated_metrics = repair_tiled(repair_base, mask_result, ai_config,
                    self.config["blend"], generate, progress=progress, cancelled=cancelled)
            else:
                patch, patch_mask, transform = extract_patch(
                    repair_base, mask_result.binary_mask, mask_result.bbox,
                    int(ai_config["context_pixels"]), int(ai_config["patch_size"]),
                )
                generated_square, generated_metrics = generate(patch, patch_mask, progress)
                generated = restore_patch(generated_square, transform)
                crop = np.s_[transform.y0:transform.y1, transform.x0:transform.x1]
                candidate = final.copy()
                candidate[crop] = blend_generated_patch(
                    repair_base[crop], generated, mask_result.binary_mask[crop],
                    mask_result.soft_mask[crop], bool(self.config["blend"]["color_match"]),
                )
                final, guard_metrics = apply_ai_guard(
                    repair_base, candidate, mask_result.binary_mask, mask_result.soft_mask,
                    self.config.get("blend", {}),
                )
                generated_metrics.update(patch_layout="single", tile_count=1, generated_tile_count=1,
                    patch_scale=(transform.inner_x1-transform.inner_x0)/transform.original_width,
                    **guard_metrics)
            neural_peak = ai_metrics.get("peak_vram_mb")
            ai_metrics = {**ai_metrics, **generated_metrics}
            if engine == "hybrid":
                ai_metrics["neural_peak_vram_mb"] = neural_peak
                ai_metrics["diffusion_peak_vram_mb"] = generated_metrics.get("peak_vram_mb")
                peaks = [p for p in (neural_peak, generated_metrics.get("peak_vram_mb")) if p is not None]
                ai_metrics["peak_vram_mb"] = max(peaks) if peaks else None
            ai_seconds = time.perf_counter() - ai_overall_started
        elif use_ai and engine == 'hybrid':
            ai_metrics.update(diffusion_skipped=True, diffusion_skip_reason=diffusion_skip_reason,
                              diffusion_policy=self.config['inpainting'].get('hybrid_diffusion_policy', 'always'),
                              hybrid_min_residual_error=float(self.config['inpainting'].get('hybrid_min_residual_error', 2.0)))
            ai_seconds = time.perf_counter() - ai_overall_started
        if cancelled and cancelled():
            raise CancelledError("已取消。")
        metrics = {
            **registration.metrics, **timings, **ai_metrics, **refinement_metrics,
            "use_ai": use_ai, "ai_seconds": ai_seconds,
            "total_seconds": time.perf_counter() - started,
            "reused_registration": prepared is not None,
            "seam_method": mask_result.method,
        }
        if use_ai and engine in ('hybrid', 'diffusion'):
            metrics['diffusion_executed'] = bool(diffusion_allowed)
            metrics['diffusion_skipped'] = not bool(diffusion_allowed)
            if engine == 'hybrid':
                metrics['diffusion_policy'] = self.config['inpainting'].get('hybrid_diffusion_policy', 'always')
        if prepared is not None:
            metrics.update({k:v for k,v in prepared.metrics.items() if k.startswith("sequence_")})
        self._notify(progress, "完成", 100)
        return StitchResult(traditional, final, registration, mask_result, metrics, poisson, allowed,
                            prepared.source_images if prepared is not None else [], repair_base)

    def run_many(self, images, *, use_ai=False, progress=None, cancelled=None):
        """Incremental planar stitching in the supplied overlap order, preserving valid pixels."""
        images = list(images)
        if len(images) < 2:
            raise StitchError("多图拼接至少需要两张图片。")
        started = time.perf_counter()
        sources = [ensure_bgr(read_image(p) if isinstance(p,(str,Path)) else p) for p in images]
        current = sources[0]
        valid = None
        accumulated_overlap = None
        accumulated_binary = None
        stages = []
        result = None
        for index, image in enumerate(sources[1:], 1):
            if cancelled and cancelled():
                raise CancelledError("已取消多图拼接。")
            def report(message, percent):
                self._notify(progress, f"第 {index}/{len(images)-1} 个接缝 · {message}",
                             int(((index-1)+percent/100)/(len(images)-1)*100))
            result = self.run(current, image, use_ai=False, progress=report,
                              cancelled=cancelled, valid_mask_a=valid)
            reg = result.registration
            h, w = result.final_image.shape[:2]
            if accumulated_overlap is not None:
                old_region = cv2.warpPerspective(accumulated_overlap,reg.canvas_transform,(w,h),flags=cv2.INTER_NEAREST)
                old_binary = cv2.warpPerspective(accumulated_binary,reg.canvas_transform,(w,h),flags=cv2.INTER_NEAREST)
                accumulated_overlap = cv2.bitwise_or(old_region,reg.overlap_mask)
                accumulated_binary = cv2.bitwise_or(old_binary,result.mask.binary_mask)
            else:
                accumulated_overlap = reg.overlap_mask.copy()
                accumulated_binary = result.mask.binary_mask.copy()
            stages.append(deepcopy(result.metrics))
            valid = cv2.bitwise_or(reg.mask_a,reg.mask_b)
            current = result.final_image
        result.repair_region = accumulated_overlap
        result.mask = mask_from_binary(accumulated_binary,accumulated_overlap,self.config["mask"]["feather_radius"],"multi_image_union")
        result.metrics.update(sequence_count=len(images), sequence_stages=stages,
                              sequence_total_seconds=time.perf_counter()-started)
        result.source_images = sources
        if use_ai:
            result = self.run(sources[0],sources[-1],prepared=result,use_ai=True,progress=progress,cancelled=cancelled)
            result.metrics["sequence_total_seconds"] = time.perf_counter()-started
        return result
