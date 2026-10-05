import numpy as np

from xiyuan_mvp.config import load_config
from xiyuan_mvp.pipeline import StitchPipeline
from xiyuan_mvp.refinement import prepare_structural_base
from tests.test_pipeline_smoke import _synthetic_pair


def test_refinement_uses_only_inputs_preserves_canvas_and_validity():
    a, b = _synthetic_pair()
    config = load_config()
    raw = StitchPipeline(config).run(a, b)
    original_a = raw.registration.canvas_a.copy()
    original_b = raw.registration.canvas_b.copy()
    base, registration, metrics = prepare_structural_base(raw.registration, config["refinement"])
    assert base.shape == raw.traditional_image.shape
    assert base.dtype == np.uint8
    np.testing.assert_array_equal(raw.registration.canvas_a, original_a)
    np.testing.assert_array_equal(raw.registration.canvas_b, original_b)
    only_a = (registration.mask_a > 0) & (registration.mask_b == 0)
    np.testing.assert_array_equal(base[only_a], original_a[only_a])
    assert not np.any(base[(registration.mask_a | registration.mask_b) == 0])
    assert metrics["refinement_enabled"] is True
    assert np.isfinite(metrics["overlap_mae_after_flow"])


def test_exposure_correction_improves_known_brightness_shift():
    a, b = _synthetic_pair()
    b = np.clip(b.astype(float) * 0.8 + 8, 0, 255).astype(np.uint8)
    config = load_config()
    raw = StitchPipeline(config).run(a, b)
    region = raw.registration.overlap_mask > 0
    before = np.abs(raw.registration.canvas_a.astype(float) - raw.registration.canvas_b)[region].mean()
    _, _, metrics = prepare_structural_base(raw.registration, {"exposure": True, "local_flow": False})
    assert metrics["overlap_mae_after_flow"] < before * 0.6
