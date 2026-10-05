from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG = Path(__file__).resolve().parent / "default.yaml"


def _merge(base: dict, overrides: dict) -> dict:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path | None = None, *, overrides: dict | None = None) -> dict[str, Any]:
    with DEFAULT_CONFIG.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if path:
        with Path(path).open("r", encoding="utf-8") as handle:
            file_overrides = yaml.safe_load(handle) or {}
        if not isinstance(file_overrides, dict):
            raise ValueError("配置文件的顶层必须是键值映射。")
        config = _merge(config, file_overrides)
    if overrides is not None:
        if not isinstance(overrides, dict):
            raise ValueError("配置覆盖值必须是键值映射。")
        config = _merge(config, deepcopy(overrides))
    validate_config(config)
    return deepcopy(config)


def validate_config(config: dict) -> None:
    ranges = {
        ("registration", "max_feature_size"): (64, 8192),
        ("registration", "ratio_test"): (0.01, 1),
        ("mask", "seam_width"): (3, 512),
        ("mask", "feather_radius"): (1, 128),
        ("inpainting", "steps"): (1, 150),
        ("inpainting", "strength"): (0.01, 1),
        ("inpainting", "guidance_scale"): (0, 30),
        ("inpainting", "context_pixels"): (0, 2048),
        ("blend", "ai_opacity"): (0, 1),
        ("blend", "ai_boundary_threshold"): (1, 4),
        ("refinement", "seam_softness"): (1, 128),
        ("refinement", "max_flow_pixels"): (0.1, 128),
        ("neural_alignment", "max_size"): (128, 1024),
        ("neural_alignment", "iterations"): (1, 32),
        ("neural_alignment", "max_flow"): (1, 128),
    }
    for (section, key), (low, high) in ranges.items():
        value = config[section][key]
        if not isinstance(value, (int, float)) or not low <= value <= high:
            raise ValueError(f"{section}.{key} 必须在 {low} 到 {high} 之间。")
    ai = config["inpainting"]
    if ai.get("hybrid_diffusion_policy", "always") not in ("always", "on_structural_gain"):
        raise ValueError("hybrid_diffusion_policy 必须是 always 或 on_structural_gain。")
    if not isinstance(ai.get("hybrid_min_residual_error", 2.0), (int, float)) or not 0 <= ai.get("hybrid_min_residual_error", 2.0) <= 128:
        raise ValueError("hybrid_min_residual_error 必须在 0 到 128 之间。")
    if ai.get("patch_layout", "single") not in ("single", "native_tiled"):
        raise ValueError("patch_layout 必须为 single 或 native_tiled。")
    tile_overlap = ai.get("tile_overlap", 128)
    if not isinstance(tile_overlap, int) or not 1 <= tile_overlap < ai["patch_size"]:
        raise ValueError("tile_overlap 必须为小于 Patch 尺寸的正整数。")
    if ai.get("controlnet_mask_mode", "none") not in ("none", "context_edges", "faded_edges"):
        raise ValueError("inpainting.controlnet_mask_mode 必须是 none、context_edges 或 faded_edges。")
    if not isinstance(ai.get("controlnet_edge_fade", 8), (int, float)) or not 1 <= ai.get("controlnet_edge_fade", 8) <= 64:
        raise ValueError("inpainting.controlnet_edge_fade 必须在 1 到 64 之间。")
    if not isinstance(ai.get("controlnet_edge_floor", 0.15), (int, float)) or not 0 <= ai.get("controlnet_edge_floor", 0.15) <= 1:
        raise ValueError("inpainting.controlnet_edge_floor 必须在 0 到 1 之间。")
    clear=config.get('clear_fusion',{})
    if clear.get('enabled') and config.get('refinement',{}).get('enabled'):
        raise ValueError('清晰融合与实验性颜色/几何校正请选择一种。')
    for key,lo,hi in [('detail_scale',1,32),('guard_threshold',0,1),('clarity_margin',1.01,4),('transition_width',2,128),('min_clarity_energy',0,100000)]:
        if key in clear and (not isinstance(clear[key],(int,float)) or not lo<=clear[key]<=hi):
            raise ValueError(f'clear_fusion.{key} 必须在 {lo} 到 {hi} 之间。')
    for key,lo,hi in [('photometric_fraction_threshold',0,1),('photometric_magnitude_threshold',0,128),('photometric_agreement_threshold',0,1)]:
        if key in clear and (not isinstance(clear[key],(int,float)) or not lo<=clear[key]<=hi):
            raise ValueError(f'clear_fusion.{key} 必须在 {lo} 到 {hi} 之间。')
    if ai.get('engine','diffusion') not in ('diffusion','neural_alignment','hybrid'):
        raise ValueError('AI engine must be diffusion, neural_alignment or hybrid.')
    if config['neural_alignment']['method'] not in ('raft','raft_large','raft_refined'):
        raise ValueError('神经修复方法必须是 raft、raft_large 或 raft_refined。')
    if ai["patch_size"] not in (256, 512, 768):
        raise ValueError("patch_size 只支持 256、512、768。")
    if int(ai["steps"] * ai["strength"]) < 1:
        raise ValueError("steps × strength 必须至少为 1，否则没有有效重绘步骤。")
