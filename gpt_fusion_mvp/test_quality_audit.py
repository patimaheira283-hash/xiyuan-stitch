"""Controlled faults exercise alerts; passing these is not real-world accuracy."""
import numpy as np
import cv2
import pytest

from .quality_audit import diagnostics, reference_comparison, detail_crop


def scene():
    y, x = np.indices((160, 240))
    gray = np.where(((x // 8 + y // 8) % 2) == 0, 60, 180).astype(np.uint8)
    base = np.repeat(gray[..., None], 3, axis=2)
    band = np.zeros((160, 240), bool)
    band[:, 75:165] = True
    return base, band


@pytest.mark.parametrize('level,code', [(0, 'new_black'), (255, 'new_white')])
def test_new_blank_region_is_flagged(level, code):
    base, band = scene()
    candidate = base.copy(); candidate[band] = level
    assert code in {a['code'] for a in diagnostics(base, candidate, band)['alerts']}


def test_preexisting_black_and_white_regions_do_not_trigger_new_blanks():
    base, band = scene()
    base[:40] = 0; base[-40:] = 255
    result = diagnostics(base, base.copy(), band)
    assert not result['alerts']
    assert result['status'] == 'review_needed'


def test_blurred_texture_is_flagged():
    base, band = scene()
    candidate = base.copy()
    candidate[band] = cv2.GaussianBlur(base, (41, 41), 10)[band]
    assert 'edge_loss' in {a['code'] for a in diagnostics(base, candidate, band)['alerts']}


def test_protected_pixel_violation_is_flagged():
    base, band = scene()
    candidate = base.copy(); candidate[0, 0] = 120
    result = diagnostics(base, candidate, band)
    assert result['metrics']['protected_pixel_changes'] == 1
    assert 'protected_change' in {a['code'] for a in result['alerts']}


def test_sharp_pattern_replacement_is_not_mistaken_for_certified_success():
    base, band = scene()
    candidate = base.copy(); candidate[band] = 240 - base[band]
    result = diagnostics(base, candidate, band)
    assert not result['alerts']  # This intentionally demonstrates a detector blind spot.
    assert result['status'] == 'review_needed'
    assert not np.array_equal(base, candidate)


def test_local_reference_score_exposes_damage_hidden_in_large_canvas():
    base, _ = scene()
    band = np.zeros(base.shape[:2], bool); band[:, 115:125] = True
    candidate = base.copy(); candidate[band] = 0
    result = reference_comparison(base, candidate, base.copy(), band)
    assert result['traditional']['seam_ssim'] == pytest.approx(1)
    assert result['candidate']['global_ssim'] > .9
    assert result['candidate']['seam_ssim'] < .3
    assert result['verdict'] == 'worse'


def test_reference_correction_can_improve_metrics():
    reference, band = scene()
    base = reference.copy(); base[band] = 0
    result = reference_comparison(base, reference.copy(), reference, band)
    assert result['verdict'] == 'better'
    assert result['candidate']['seam_ssim'] == pytest.approx(1)


def test_wrong_canvas_and_empty_mask_are_rejected():
    base, band = scene()
    with pytest.raises(ValueError, match='exact canvas'):
        reference_comparison(base, base.copy(), base[:-1], band)
    with pytest.raises(ValueError, match='nonempty'):
        diagnostics(base, base.copy(), np.zeros_like(band))


def test_detail_crop_covers_local_change_and_stays_within_canvas():
    base, band = scene()
    candidate = base.copy(); candidate[100:120, 100:120] = 0
    left, top, right, bottom = detail_crop(base, candidate, band)
    assert 0 <= left <= 100 < 120 <= right <= base.shape[1]
    assert 0 <= top <= 100 < 120 <= bottom <= base.shape[0]
