from copy import deepcopy
import json

import cv2
import numpy as np
import pytest

from xiyuan_mvp.benchmark import masked_metrics, run_benchmark
from xiyuan_mvp.config import load_config
from xiyuan_mvp.errors import InpaintingUnavailableError, StitchError
from xiyuan_mvp.image_io import write_image
from xiyuan_mvp.pipeline import StitchPipeline
from xiyuan_mvp.run_io import load_run, save_run, write_json
from tests.test_pipeline_smoke import _synthetic_pair


@pytest.fixture
def prepared():
    a, b = _synthetic_pair()
    config = load_config()
    return a, b, config, StitchPipeline(config).run(a, b)


def test_run_archive_roundtrip_and_hash_validation(tmp_path, prepared):
    a, b, config, result = prepared
    save_run(tmp_path, result, config)
    loaded, report = load_run(tmp_path)
    np.testing.assert_array_equal(loaded.poisson_image, result.poisson_image)
    np.testing.assert_array_equal(loaded.mask.binary_mask, result.mask.binary_mask)
    assert loaded.metrics["use_ai"] is False
    assert "07_ai_result" not in report["artifacts"]
    (tmp_path / "input_a.png").write_bytes(b"modified")
    with pytest.raises(ValueError, match="校验值"):
        load_run(tmp_path)


def test_custom_mask_cannot_silently_resize(prepared):
    a, b, config, result = prepared
    with pytest.raises(StitchError, match="尺寸"):
        StitchPipeline(config).run(a, b, prepared=result, custom_mask=np.ones((10, 10), np.uint8))


@pytest.mark.parametrize('executed', [False, True])
def test_desktop_reports_whether_diffusion_really_ran(prepared, executed):
    from PySide6.QtWidgets import QApplication
    from xiyuan_mvp.gui import MainWindow
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    result = prepared[3]
    result.metrics.update(use_ai=True, diffusion_executed=executed, diffusion_skipped=not executed)
    try:
        window._on_result(result)
        text = window.status_label.text()
        assert ('已运行生成修复' if executed else '未运行生成修复') in text
    finally:
        window.close()


@pytest.mark.parametrize("opacity", [0, 0.2, 1])
def test_prepared_ai_reuses_geometry_and_current_parameters(monkeypatch, prepared, opacity):
    a, b, config, result = prepared
    from xiyuan_mvp.inpainting import DiffusersInpainter
    seeds = []
    def generate(self, image, mask, **kwargs):
        seeds.append(self.config["seed"])
        output = image.copy()
        output[mask > 0] = 220
        self.last_metrics = {"test_double": True}
        return output
    monkeypatch.setattr(DiffusersInpainter, "generate", generate)
    def forbidden(*args, **kwargs):
        raise AssertionError("must not re-register an edited canvas")
    monkeypatch.setattr("xiyuan_mvp.pipeline.register_images", forbidden)
    pipe = StitchPipeline(config)
    pipe.config["blend"]["ai_opacity"] = opacity
    for seed in [7, 13]:
        pipe.config["inpainting"]["seed"] = seed
        generated = pipe.run(a, b, prepared=result, custom_mask=result.mask.binary_mask, use_ai=True)
        outside = result.mask.soft_mask == 0
        np.testing.assert_array_equal(generated.final_image[outside], result.traditional_image[outside])
        if opacity == 0:
            np.testing.assert_array_equal(generated.final_image, result.traditional_image)
    assert seeds == [7, 13]


def test_benchmark_keeps_baselines_when_ai_fails(tmp_path, monkeypatch):
    from xiyuan_mvp.inpainting import DiffusersInpainter
    def unavailable(*args, **kwargs):
        raise InpaintingUnavailableError("GPU unavailable in test")
    monkeypatch.setattr(DiffusersInpainter, "generate", unavailable)
    a, b = _synthetic_pair()
    write_image(tmp_path / "a.png", a)
    write_image(tmp_path / "b.png", b)
    write_json(tmp_path / "manifest.json", {"cases": [
        {"id": "case_1", "image_a": "a.png", "image_b": "b.png", "source_id": "one"}
    ]})
    summary = run_benchmark(tmp_path / "manifest.json", tmp_path / "out", load_config(), use_ai=True)
    assert summary["registration_successes"] == 1
    assert summary["ai_successes"] == 0
    assert summary["ai_failures"] == 1
    assert (tmp_path / "out/case_1/06_poisson.png").is_file()
    assert not (tmp_path / "out/case_1/07_ai_result.png").exists()
    assert summary["records"][0]["quality"] is None
    assert "GPU unavailable" in (tmp_path / "out/case_1/error.json").read_text()


def test_metric_region_and_identical_result():
    ref = np.full((64, 64, 3), 100, np.uint8)
    mask = np.zeros((64, 64), np.uint8)
    mask[20:40, 20:40] = 255
    result = ref.copy()
    result[:10] = 0
    metrics = masked_metrics(result, ref, mask)
    assert metrics["mae"] == 0
    assert metrics["ssim"] == pytest.approx(1)
    result[mask > 0] = 110
    assert masked_metrics(result, ref, mask)["mae"] == 10


def test_config_partial_override_is_installable(tmp_path):
    override = tmp_path / "config.yaml"
    override.write_text("inpainting:\n  seed: 42\n", encoding="utf-8")
    config = load_config(override)
    assert config["inpainting"]["seed"] == 42
    assert config["registration"]["min_matches"] == 12


def test_gui_input_change_invalidates_mask_and_ai(tmp_path, prepared):
    from PySide6.QtWidgets import QApplication
    from xiyuan_mvp.gui import MainWindow
    app = QApplication.instance() or QApplication([])
    a, b, config, result = prepared
    path = tmp_path / "new.png"
    write_image(path, a)
    window = MainWindow(output_root=tmp_path / "runs")
    window._on_result(result)
    assert window.mask_view.mask is not None
    window.set_input("a", str(path))
    assert window.result is None
    assert window.mask_view.mask is None
    assert not window.ai_button.isEnabled()
    assert not window.save_button.isEnabled()
    window.close()


def test_three_image_pipeline_preserves_all_seams_and_sources(tmp_path):
    a,b=_synthetic_pair()
    images=[a[:,:600],a[:,360:],b[:,200:]]
    result=StitchPipeline().run_many(images)
    assert result.metrics["sequence_count"]==3
    assert result.final_image.shape[1]>=1150
    assert len(result.source_images)==3
    assert np.any((result.repair_region>0)&(result.registration.overlap_mask==0))
    assert np.any((result.mask.binary_mask>0)&(result.registration.overlap_mask==0))
    save_run(tmp_path,result,load_config())
    restored,_=load_run(tmp_path)
    assert len(restored.source_images)==3
    np.testing.assert_array_equal(restored.repair_region,result.repair_region)


def test_controlnet_conditioning_keeps_geometry():
    from xiyuan_mvp.inpainting import control_image
    image=np.zeros((128,256,3),np.uint8)
    image[:,128:]=255
    canny=np.asarray(control_image(image,"canny",{}))
    tile=np.asarray(control_image(image,"tile",{}))
    assert canny.shape==image.shape
    assert 0<np.count_nonzero(canny)<128*256
    np.testing.assert_array_equal(tile,image[:,:,::-1])


def test_controlnet_context_edges_suppress_only_the_repair_region():
    from xiyuan_mvp.inpainting import control_image
    image = np.zeros((96, 160, 3), np.uint8)
    image[:, 80:] = 255
    image[20:76, 80:82] = 0
    mask = np.zeros((96, 160), np.uint8)
    mask[30:66, 74:88] = 255
    raw = np.asarray(control_image(image, "canny", {"controlnet_mask_mode": "none"}))
    context = np.asarray(control_image(image, "canny", {"controlnet_mask_mode": "context_edges"}, mask))
    assert np.count_nonzero(context[mask > 0]) == 0
    np.testing.assert_array_equal(context[mask == 0], raw[mask == 0])
    assert np.count_nonzero(raw[mask > 0]) > 0


def test_controlnet_faded_edges_are_bounded_and_tile_is_unchanged():
    from xiyuan_mvp.inpainting import control_image
    image = np.zeros((64, 128, 3), np.uint8)
    image[:, 64:] = 255
    mask = np.zeros((64, 128), np.uint8)
    mask[16:48, 56:72] = 255
    config = {
        "controlnet_mask_mode": "faded_edges",
        "controlnet_edge_fade": 6,
        "controlnet_edge_floor": 0.2,
    }
    raw = np.asarray(control_image(image, "canny", {"controlnet_mask_mode": "none"}, mask))
    faded = np.asarray(control_image(image, "canny", config, mask))
    assert np.all(faded <= raw)
    assert np.count_nonzero(faded) <= np.count_nonzero(raw)
    np.testing.assert_array_equal(
        np.asarray(control_image(image, "tile", config, mask)), image[:, :, ::-1]
    )


def test_gui_exposes_canny_mask_modes_only_for_canny(tmp_path):
    from PySide6.QtWidgets import QApplication
    from xiyuan_mvp.gui import MainWindow
    app = QApplication.instance() or QApplication([])
    window = MainWindow(output_root=tmp_path / "runs")
    window.repair_engine.setCurrentIndex(window.repair_engine.findData("diffusion"))
    window.controlnet.setCurrentText("none")
    assert not window.controlnet_mask_mode.isEnabled()
    window.controlnet.setCurrentText("canny")
    assert window.controlnet_mask_mode.isEnabled()
    window.controlnet.setCurrentText("tile")
    assert not window.controlnet_mask_mode.isEnabled()
    window.close()


def test_open_run_clears_previous_input_sequence(tmp_path, prepared, monkeypatch):
    from PySide6.QtWidgets import QApplication, QFileDialog, QListWidgetItem
    from xiyuan_mvp.gui import MainWindow
    app = QApplication.instance() or QApplication([])
    a,b,config,result=prepared
    save_run(tmp_path/'record',result,config)
    window=MainWindow()
    window.sequence_list.addItem(QListWidgetItem('unrelated-old-photo'))
    monkeypatch.setattr(QFileDialog,'getOpenFileName',lambda *a,**k:(str(tmp_path/'record/run.json'),'JSON'))
    window._open_run()
    assert window.result is not None
    assert window.sequence_list.count()==0
    window.close()


def test_open_legacy_run_merges_nested_defaults(tmp_path, prepared, monkeypatch):
    from PySide6.QtWidgets import QApplication, QFileDialog
    from xiyuan_mvp.gui import MainWindow
    from xiyuan_mvp.config import validate_config
    app = QApplication.instance() or QApplication([])
    _, _, config, result = prepared
    legacy = deepcopy(config)
    del legacy["blend"]["ai_boundary_threshold"]
    del legacy["inpainting"]["tile_overlap"]
    legacy["blend"]["ai_opacity"] = 0.37
    legacy["inpainting"]["seed"] = 123
    save_run(tmp_path / "legacy", result, legacy)
    before = (tmp_path / "legacy/run.json").read_bytes()
    window = MainWindow()
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        lambda *a, **k: (str(tmp_path / "legacy/run.json"), "JSON"))
    try:
        window._open_run()
        assert window.result is not None
        window._sync_config()
        validate_config(window.config)
        assert window.config["blend"]["ai_boundary_threshold"] == config["blend"]["ai_boundary_threshold"]
        assert window.config["inpainting"]["tile_overlap"] == config["inpainting"]["tile_overlap"]
        assert window.config["blend"]["ai_opacity"] == pytest.approx(0.37)
        assert window.config["inpainting"]["seed"] == 123
        assert (tmp_path / "legacy/run.json").read_bytes() == before
    finally:
        window.close()
