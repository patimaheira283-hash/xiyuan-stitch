from __future__ import annotations

from dataclasses import dataclass
import gc
import time

import cv2
import numpy as np
from PIL import Image

from .errors import CancelledError, InpaintingUnavailableError

CONTROLNET_MODELS = {
    "canny": ("lllyasviel/control_v11p_sd15_canny", "115a470d547982438f70198e353a921996e2e819", True),
    "tile": ("lllyasviel/control_v11f1e_sd15_tile", "3f877705c37010b7221c3d10743307d6b5b6efac", False),
}


def _control_mask(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
    if mask is None:
        return None
    if mask.shape != shape:
        raise ValueError("ControlNet mask 尺寸必须与控制图一致。")
    return (mask > 0).astype(np.uint8)


def control_image(
    image_bgr: np.ndarray,
    mode: str,
    config: dict,
    mask: np.ndarray | None = None,
) -> Image.Image:
    """Build the ControlNet conditioning image for an inpainting patch.

    ``controlnet_mask_mode`` is deliberately opt-in.  The old behavior is
    ``none``: Canny sees the complete patch.  ``context_edges`` removes edges
    inside the region that will be regenerated, while ``faded_edges`` keeps a
    small amount of that signal and fades it across the mask boundary.
    """
    if mode == "canny":
        edges = cv2.Canny(image_bgr, int(config.get("canny_low", 100)), int(config.get("canny_high", 200)))
        mask_binary = _control_mask(mask, edges.shape)
        mask_mode = str(config.get("controlnet_mask_mode", "none"))
        if mask_binary is not None and mask_mode != "none":
            if mask_mode == "context_edges":
                edges = np.where(mask_binary > 0, 0, edges).astype(np.uint8)
            elif mask_mode == "faded_edges":
                radius = max(1, int(config.get("controlnet_edge_fade", 8)))
                kernel = radius * 2 + 1
                softened = cv2.GaussianBlur(mask_binary.astype(np.float32), (kernel, kernel), 0)
                floor = float(np.clip(config.get("controlnet_edge_floor", 0.15), 0.0, 1.0))
                alpha = 1.0 - (softened * (1.0 - floor))
                edges = np.rint(edges.astype(np.float32) * alpha).astype(np.uint8)
            else:
                raise ValueError(
                    "controlnet_mask_mode 必须是 none、context_edges 或 faded_edges。"
                )
        return Image.fromarray(np.repeat(edges[:, :, None], 3, axis=2))
    return Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))


@dataclass(slots=True)
class PatchTransform:
    x0: int
    y0: int
    x1: int
    y1: int
    original_width: int
    original_height: int
    inner_x0: int
    inner_y0: int
    inner_x1: int
    inner_y1: int


def extract_patch(
    image: np.ndarray,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    context_pixels: int,
    patch_size: int,
) -> tuple[np.ndarray, np.ndarray, PatchTransform]:
    height, width = image.shape[:2]
    x0, y0, x1, y1 = bbox
    x0 = max(0, x0 - context_pixels)
    y0 = max(0, y0 - context_pixels)
    x1 = min(width, x1 + context_pixels)
    y1 = min(height, y1 + context_pixels)
    crop = image[y0:y1, x0:x1]
    crop_mask = mask[y0:y1, x0:x1]
    crop_height, crop_width = crop.shape[:2]
    scale = min(patch_size / crop_width, patch_size / crop_height)
    resized_width = max(1, int(round(crop_width * scale)))
    resized_height = max(1, int(round(crop_height * scale)))
    inner_x0 = (patch_size - resized_width) // 2
    inner_y0 = (patch_size - resized_height) // 2
    inner_x1 = inner_x0 + resized_width
    inner_y1 = inner_y0 + resized_height
    transform = PatchTransform(
        x0,
        y0,
        x1,
        y1,
        crop_width,
        crop_height,
        inner_x0,
        inner_y0,
        inner_x1,
        inner_y1,
    )
    resized_crop = cv2.resize(
        crop, (resized_width, resized_height), interpolation=cv2.INTER_AREA
    )
    resized_crop_mask = cv2.resize(
        crop_mask, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST
    )
    top = inner_y0
    bottom = patch_size - inner_y1
    left = inner_x0
    right = patch_size - inner_x1
    padded_image = cv2.copyMakeBorder(
        resized_crop, top, bottom, left, right, cv2.BORDER_REFLECT_101
    )
    padded_mask = cv2.copyMakeBorder(
        resized_crop_mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0
    )
    return padded_image, padded_mask, transform


def restore_patch(patch: np.ndarray, transform: PatchTransform) -> np.ndarray:
    content = patch[
        transform.inner_y0 : transform.inner_y1,
        transform.inner_x0 : transform.inner_x1,
    ]
    return cv2.resize(
        content,
        (transform.original_width, transform.original_height),
        interpolation=cv2.INTER_CUBIC,
    )


class DiffusersInpainter:
    def __init__(self, config: dict):
        self.config = config
        self._pipeline = None
        self.device = "cpu"
        self._load_key = None
        self.last_metrics: dict = {}

    def _load(self) -> None:
        try:
            import torch
            from diffusers import StableDiffusionInpaintPipeline, StableDiffusionControlNetInpaintPipeline, ControlNetModel, LCMScheduler
        except ImportError as exc:
            raise InpaintingUnavailableError('AI 依赖未安装，请运行 pip install -e ".[ai]"。') from exc
        requested = str(self.config.get("device", "auto"))
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else requested
        if self.device not in ("cpu", "cuda"):
            raise InpaintingUnavailableError("当前支持 cpu 或 cuda 推理设备。")
        if self.device == "cpu" and not bool(self.config.get("allow_cpu", False)):
            raise InpaintingUnavailableError("未检测到 CUDA GPU。请使用免费 GPU 实验包，或在设置中启用较慢的 CPU 推理。")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise InpaintingUnavailableError("配置要求 CUDA，但当前 PyTorch 未检测到可用 NVIDIA GPU。")
        if self.device == "cpu":
            torch.set_num_threads(int(self.config.get("cpu_threads", 8)))
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        try:
            mode = self.config.get("controlnet", "none")
            options = {}
            pipeline_type = StableDiffusionInpaintPipeline
            if mode in CONTROLNET_MODELS:
                model_id, revision, safe_weights = CONTROLNET_MODELS[mode]
                options["controlnet"] = ControlNetModel.from_pretrained(
                    model_id, revision=revision, torch_dtype=dtype,
                    variant="fp16" if safe_weights else None,
                    use_safetensors=safe_weights,
                    local_files_only=bool(self.config.get("local_files_only", True)),
                    cache_dir=self.config.get("cache_dir", "models/hub"),
                )
                pipeline_type = StableDiffusionControlNetInpaintPipeline
            elif mode != "none":
                raise InpaintingUnavailableError("ControlNet 模式必须为 none、canny 或 tile。")
            pipeline = pipeline_type.from_pretrained(
                self.config["model_id"], torch_dtype=dtype,
                revision=self.config.get("revision", "main"),
                variant=self.config.get("variant"),
                cache_dir=self.config.get("cache_dir", "models/hub"),
                local_files_only=bool(self.config.get("local_files_only", True)),
                use_safetensors=True,
                **options,
            )
            if self.config.get("use_lcm", False):
                pipeline.load_lora_weights(
                    self.config.get("lcm_model_id", "latent-consistency/lcm-lora-sdv1-5"),
                    revision=self.config.get("lcm_revision", "cf2fced511dbe7e26c8d1d397e728fbab875db4b"),
                    weight_name="pytorch_lora_weights.safetensors",
                    local_files_only=bool(self.config.get("local_files_only", True)),
                    cache_dir=self.config.get("cache_dir", "models/hub"),
                )
                pipeline.fuse_lora()
                pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
            pipeline.to(self.device)
            pipeline.enable_attention_slicing()
            pipeline.enable_vae_slicing()
            pipeline.set_progress_bar_config(disable=True)
            self._pipeline = pipeline
        except OSError as exc:
            raise InpaintingUnavailableError(
                "模型无法加载。请先运行 python -m xiyuan_mvp.model_setup 下载模型，"
                "或在设置中指定完整本地模型目录。详情：" + str(exc)
            ) from exc

    def generate(self, image_bgr: np.ndarray, mask: np.ndarray, *, progress=None, cancelled=None) -> np.ndarray:
        load_started = time.perf_counter()
        key = tuple(str(self.config.get(k)) for k in (
            "model_id", "revision", "variant", "device", "allow_cpu", "cache_dir", "local_files_only",
            "controlnet", "use_lcm", "lcm_model_id", "lcm_revision",
        ))
        if self._pipeline is None or self._load_key != key:
            self._pipeline = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            if progress:
                progress("加载扩散模型（首次运行较慢）", 72)
            self._load()
            self._load_key = key
        load_seconds = time.perf_counter() - load_started
        import torch
        seed = int(self.config.get("seed", 2026))
        generator = torch.Generator(device=self.device).manual_seed(seed)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        steps = int(self.config.get("steps", 20))
        strength = float(self.config.get("strength", 0.5))
        mode = self.config.get("controlnet", "none")
        control_options = {}
        if mode in CONTROLNET_MODELS:
            control_options = {
                "control_image": control_image(image_bgr, mode, self.config, mask=mask),
                "controlnet_conditioning_scale": float(self.config.get("controlnet_scale", 0.65)),
            }

        def on_step(_pipeline, index, _timestep, callback_kwargs):
            if cancelled and cancelled():
                raise CancelledError("已取消 AI 修复，保留传统结果。")
            if progress:
                total = max(int(steps * strength), 1)
                progress(f"AI 修复 {min(index + 1, total)}/{total} 步", 78 + int(17 * min((index + 1) / total, 1)))
            return callback_kwargs

        started = time.perf_counter()
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        try:
            if cancelled and cancelled():
                raise CancelledError("已取消 AI 修复。")
            with torch.inference_mode():
                output = self._pipeline(
                    prompt=str(self.config.get("prompt", "")),
                    negative_prompt=str(self.config.get("negative_prompt", "")),
                    image=Image.fromarray(image_rgb), mask_image=Image.fromarray(mask),
                    height=image_bgr.shape[0], width=image_bgr.shape[1],
                    num_inference_steps=steps,
                    guidance_scale=float(self.config.get("guidance_scale", 5.0)),
                    strength=strength, generator=generator,
                    callback_on_step_end=on_step,
                    **control_options,
                )
            if getattr(output, "nsfw_content_detected", None) and any(output.nsfw_content_detected):
                raise InpaintingUnavailableError("模型内容检查未通过，本次生成结果未保存。")
            self.last_metrics = {
                "device": self.device,
                "device_name": torch.cuda.get_device_name(0) if self.device == "cuda" else "CPU",
                "model_id": self.config["model_id"],
                "model_revision": self.config.get("revision", "main"),
                "model_variant": self.config.get("variant"),
                "controlnet": mode,
                "controlnet_mask_mode": self.config.get("controlnet_mask_mode", "none"),
                "controlnet_model": CONTROLNET_MODELS[mode][0] if mode in CONTROLNET_MODELS else None,
                "controlnet_revision": CONTROLNET_MODELS[mode][1] if mode in CONTROLNET_MODELS else None,
                "use_lcm": bool(self.config.get("use_lcm", False)),
                "lcm_revision": self.config.get("lcm_revision") if self.config.get("use_lcm") else None,
                "effective_steps": max(int(steps * strength), 1),
                "model_load_seconds": load_seconds,
                "inference_seconds": time.perf_counter() - started,
                "peak_vram_mb": torch.cuda.max_memory_allocated() / 1024**2 if self.device == "cuda" else None,
                "peak_reserved_vram_mb": torch.cuda.max_memory_reserved() / 1024**2 if self.device == "cuda" else None,
            }
            return cv2.cvtColor(np.asarray(output.images[0]), cv2.COLOR_RGB2BGR)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise InpaintingUnavailableError("显存或内存不足。请使用 512 Patch、关闭其他占用显存的程序，或在 GPU 环境运行。") from exc
            raise
        finally:
            if self.device == "cuda":
                torch.cuda.empty_cache()
