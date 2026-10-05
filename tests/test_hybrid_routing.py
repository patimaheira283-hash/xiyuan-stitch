import cv2
import numpy as np

from xiyuan_mvp.config import load_config
from xiyuan_mvp.inpainting import DiffusersInpainter
from xiyuan_mvp.pipeline import StitchPipeline
from tests.test_pipeline_smoke import _synthetic_pair


def test_hybrid_passes_protected_raft_result_into_diffusion(monkeypatch):
    first, second = _synthetic_pair()
    config = load_config()
    config["inpainting"].update(engine="hybrid", controlnet="none")
    config["blend"]["ai_opacity"] = 1.0
    pipe = StitchPipeline(config)
    seen = {}

    def fake_raft(registration, base, mask, neural_config, *, method):
        repaired = base.copy()
        repaired[mask.binary_mask > 0] = (base[mask.binary_mask > 0].astype(np.int16) + 7).clip(0, 255).astype(np.uint8)
        return repaired, {"repair_engine": "raft_large", "repair_applied": True, "model_executed": True}

    def fake_generate(self, image, mask, **kwargs):
        seen["patch_mean"] = float(image[mask > 0].mean())
        output = image.copy()
        output[mask > 0] = np.clip(output[mask > 0].astype(np.int16) + 3, 0, 255).astype(np.uint8)
        self.last_metrics = {"device": "test", "test_double": False, "inference_seconds": 0.01}
        return output

    monkeypatch.setattr("xiyuan_mvp.neural_alignment.repair_alignment", fake_raft)
    monkeypatch.setattr(DiffusersInpainter, "generate", fake_generate)
    result = pipe.run(first, second, use_ai=True)
    assert result.metrics["repair_engine"] == "raft_large"
    assert result.metrics["model_executed"] is True
    assert result.metrics["use_ai"] is True
    assert result.metrics["test_double"] is False
    assert "patch_mean" in seen
    outside = result.mask.soft_mask == 0
    np.testing.assert_array_equal(result.final_image[outside], result.repair_base_image[outside])
    assert np.any(result.final_image[result.mask.binary_mask > 0] != result.repair_base_image[result.mask.binary_mask > 0])


def test_hybrid_gate_skips_generation_without_confirmed_structural_gain(monkeypatch):
    first, second = _synthetic_pair()
    config = load_config()
    config["inpainting"].update(engine="hybrid", hybrid_diffusion_policy="on_structural_gain")
    pipe = StitchPipeline(config)
    def fake_raft(registration, base, mask, neural_config, *, method):
        return base.copy(), {"repair_engine": "raft_large", "repair_applied": False,
                             "model_executed": False, "repair_reason": "clear_source_already_selected"}
    def forbidden(*args, **kwargs):
        raise AssertionError("门控策略不应在没有结构收益时启动扩散")
    monkeypatch.setattr("xiyuan_mvp.neural_alignment.repair_alignment", fake_raft)
    monkeypatch.setattr(DiffusersInpainter, "generate", forbidden)
    result = pipe.run(first, second, use_ai=True)
    assert result.metrics["diffusion_skipped"] is True
    assert result.metrics["diffusion_skip_reason"] == "no_confirmed_structural_gain"
    assert result.metrics["diffusion_policy"] == "on_structural_gain"
