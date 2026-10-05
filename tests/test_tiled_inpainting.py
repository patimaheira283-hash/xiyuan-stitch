from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from xiyuan_mvp.config import load_config
from xiyuan_mvp.errors import CancelledError
from xiyuan_mvp.pipeline import StitchPipeline
from xiyuan_mvp.seam_mask import mask_from_binary
from xiyuan_mvp.tiled_inpainting import plan_tiles, repair_tiled
from tests.test_clear_fusion import pair


def masks(shape, *, sparse=False):
    binary = np.zeros(shape, np.uint8)
    if sparse:
        binary[:30, :30] = 255
        binary[-30:, -30:] = 255
    else:
        binary[:, shape[1]//2-30:shape[1]//2+30] = 255
    return mask_from_binary(binary, np.full(shape, 255, np.uint8), 8)


def config():
    return {"patch_size": 256, "tile_overlap": 64, "context_pixels": 64}


@pytest.mark.parametrize("shape,sparse", [((1800, 360), False), ((320, 4000), True), ((31, 79), False)])
def test_identity_native_tiles_preserve_every_pixel_including_fine_texture(shape, sparse):
    base = np.random.default_rng(11).integers(0, 256, (*shape, 3), np.uint8)
    before = base.copy()
    mask = masks(shape, sparse=sparse)
    calls = []
    def generate(patch, binary, callback):
        assert patch.shape == (256, 256, 3) and binary.shape == (256, 256)
        calls.append(1)
        return patch.copy(), {"inference_seconds": 1., "peak_vram_mb": 512.}
    result, info = repair_tiled(base, mask, config(), {"color_match": False}, generate)
    np.testing.assert_array_equal(result, before)
    np.testing.assert_array_equal(base, before)
    assert info["patch_scale"] == 1.
    assert info["inference_seconds"] == len(calls)
    assert info["peak_vram_mb"] == 512.
    if sparse:
        assert len(calls) < 10  # The gap between distant seams is not generated.


def test_overlaps_apply_opacity_once_and_report_max_memory():
    base = np.full((920, 360, 3), 80, np.uint8)
    mask = masks(base.shape[:2])
    seen = []
    def generate(patch, binary, callback):
        # Later patches must still see the original base, not earlier predictions.
        assert np.all(patch == 80)
        seen.append(1)
        return np.full_like(patch, 160), {"inference_seconds": 2., "peak_vram_mb": len(seen)*100.}
    result, info = repair_tiled(base, mask, config(), {"color_match": False, "ai_opacity": .5}, generate)
    assert len(seen) > 1
    assert np.all(result[mask.soft_mask == 255] == 120)
    np.testing.assert_array_equal(result[mask.soft_mask == 0], base[mask.soft_mask == 0])
    assert info["inference_seconds"] == len(seen)*2
    assert info["peak_vram_mb"] == len(seen)*100


@pytest.mark.parametrize("cancel", [False, True])
def test_failed_or_cancelled_tile_never_mutates_base(cancel):
    base = np.full((920, 360, 3), 80, np.uint8)
    original = base.copy()
    count = [0]
    def generate(patch, binary, callback):
        count[0] += 1
        if not cancel and count[0] == 2:
            raise RuntimeError("tile failed")
        return np.zeros_like(patch), {}
    with pytest.raises(CancelledError if cancel else RuntimeError):
        repair_tiled(base, masks(base.shape[:2]), config(), {}, generate,
                     cancelled=lambda: cancel and count[0] >= 1)
    np.testing.assert_array_equal(base, original)


def test_pipeline_routes_native_tiles_without_resizing_or_reregistering(monkeypatch):
    reg, _ = pair("b")
    cfg = load_config()
    cfg["inpainting"].update(patch_layout="native_tiled", patch_size=256, tile_overlap=64)
    cfg["blend"].update(color_match=False)
    monkeypatch.setattr("xiyuan_mvp.pipeline.register_images", lambda *a, **k: reg)
    pipe = StitchPipeline(cfg)
    baseline = pipe.run(reg.image_a, reg.image_b)
    before = baseline.final_image.copy()
    class Identity:
        def generate(self, image, mask): return image.copy()
    pipe._inpainter = Identity()
    def forbidden(*args, **kwargs): raise AssertionError("unexpected resampling/registration")
    monkeypatch.setattr("xiyuan_mvp.pipeline.register_images", forbidden)
    monkeypatch.setattr("xiyuan_mvp.pipeline.extract_patch", forbidden)
    result = pipe.run(reg.image_a, reg.image_b, prepared=baseline, use_ai=True)
    np.testing.assert_array_equal(result.final_image, before)
    assert result.metrics["patch_layout"] == "native_tiled"
    assert result.metrics["test_double"]
