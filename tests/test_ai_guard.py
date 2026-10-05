import numpy as np
import pytest

from xiyuan_mvp.ai_guard import apply_ai_guard, boundary_discontinuity


def _case():
    base = np.full((32, 32, 3), 128, np.uint8)
    candidate = base.copy()
    candidate[14:18, 8:24] = 255
    binary = np.zeros((32, 32), np.uint8)
    binary[14:18, 8:24] = 255
    soft = binary.copy()
    return base, candidate, binary, soft


def test_guard_reduces_a_boundary_regression():
    base, candidate, binary, soft = _case()
    output, metrics = apply_ai_guard(
        base, candidate, binary, soft,
        {"ai_opacity": 1.0, "ai_boundary_guard": True, "ai_boundary_threshold": 1.05},
    )
    assert metrics["ai_boundary_guard_triggered"]
    assert metrics["ai_effective_opacity"] < 1.0
    assert boundary_discontinuity(output, binary) < boundary_discontinuity(candidate, binary)
    assert np.array_equal(output[soft == 0], base[soft == 0])


def test_guard_off_preserves_requested_opacity():
    base, candidate, binary, soft = _case()
    output, metrics = apply_ai_guard(
        base, candidate, binary, soft,
        {"ai_opacity": 0.25, "ai_boundary_guard": False},
    )
    assert not metrics["ai_boundary_guard_triggered"]
    assert metrics["ai_effective_opacity"] == 0.25
    assert int(output[15, 10, 0]) == 160


def test_already_feathered_candidate_is_not_feathered_twice():
    base = np.full((32, 32, 3), 80, np.uint8)
    candidate = np.full_like(base, 120)  # 160 generated at approximately 50% feather
    mask = np.full((32, 32), 255, np.uint8)
    soft = np.full_like(mask, 128)
    out, metrics = apply_ai_guard(base, candidate, mask, soft,
                                 {"ai_opacity": .5, "ai_boundary_guard": False})
    np.testing.assert_array_equal(out, np.full_like(base, 100))
    assert metrics["ai_opacity"] == .5


def test_boundary_score_ignores_texture_far_inside_mask():
    base = np.full((96, 96, 3), 128, np.uint8)
    textured = base.copy()
    textured[36:60, 36:60] = np.random.default_rng(67).integers(0, 256, (24, 24, 3), np.uint8)
    mask = np.zeros((96, 96), np.uint8)
    mask[12:84, 12:84] = 255
    assert boundary_discontinuity(textured, mask) == boundary_discontinuity(base, mask)


@pytest.mark.parametrize('layout', ['single', 'native_tiled'])
def test_pipeline_partial_mask_applies_feather_and_opacity_once(monkeypatch, layout):
    from xiyuan_mvp.config import load_config
    from xiyuan_mvp.pipeline import StitchPipeline
    from xiyuan_mvp.types import MaskResult
    from tests.test_clear_fusion import pair

    reg, _ = pair(None)
    monkeypatch.setattr('xiyuan_mvp.pipeline.register_images', lambda *a, **k: reg)
    cfg = load_config()
    cfg['inpainting'].update(engine='diffusion', patch_layout=layout, patch_size=256,
                              tile_overlap=64, context_pixels=0)
    cfg['blend'].update(color_match=False, ai_opacity=.5, ai_boundary_guard=False)
    pipe = StitchPipeline(cfg)
    prepared = pipe.run(reg.image_a, reg.image_b)
    prepared.repair_base_image = np.full_like(prepared.final_image, 80)
    binary = np.zeros_like(reg.mask_a)
    binary[50:130, 160:240] = 255
    soft = np.where(binary > 0, 128, 0).astype(np.uint8)
    prepared.mask = MaskResult(binary, soft, (160, 50, 240, 130))
    class ConstantPrediction:
        def generate(self, image, mask):
            return np.full_like(image, 160)
    pipe._inpainter = ConstantPrediction()
    result = pipe.run(reg.image_a, reg.image_b, prepared=prepared, use_ai=True)
    assert np.all(result.final_image[binary > 0] == 100)
    assert np.all(result.final_image[binary == 0] == 80)
