from dataclasses import replace
import cv2
import numpy as np
import pytest
from xiyuan_mvp.clear_fusion import fuse, low_frequency_photometric_disagreement
from xiyuan_mvp.blending import feather_blend
from xiyuan_mvp.config import load_config
from xiyuan_mvp.pipeline import StitchPipeline
from xiyuan_mvp.types import RegistrationResult


def test_clear_fusion_is_enabled_in_the_shipped_default():
    config = load_config()['clear_fusion']
    assert config['enabled'] is True
    assert config['clarity_margin'] == 1.05


def test_clear_fusion_revision_includes_photometric_guard():
    _, metrics = fuse(pair(None)[0], {'mode': 'adaptive_detail', 'guard': True})
    assert metrics['algorithm_revision'] == 'adaptive-noise-photometric-v4'


def test_low_texture_guard_keeps_neutral_blend():
    reg, _ = pair('b')
    result, metrics = fuse(reg, {'mode': 'adaptive_detail', 'guard': True,
                                 'clarity_margin': 1.01, 'min_clarity_energy': 999999})
    assert metrics['low_texture_guard_applied']
    assert metrics['selection'] == 'balanced; preserve feather'
    np.testing.assert_array_equal(result, feather_blend(reg))


def pair(blur_side):
    rng=np.random.default_rng(177)
    scene=cv2.GaussianBlur(rng.integers(0,256,(180,400,3),np.uint8),(0,0),.6)
    a=np.zeros_like(scene);b=a.copy();ma=np.zeros(scene.shape[:2],np.uint8);mb=ma.copy()
    a[:,:280]=scene[:,:280];b[:,120:]=scene[:,120:];ma[:,:280]=255;mb[:,120:]=255
    if blur_side=='a':a[:,:280]=cv2.GaussianBlur(scene[:,:280],(0,0),1.3)
    if blur_side=='b':b[:,120:]=cv2.GaussianBlur(scene[:,120:],(0,0),1.3)
    return RegistrationResult(a[:,:280],b[:,120:],a,b,ma,mb,cv2.bitwise_and(ma,mb),np.eye(3),np.eye(3),np.zeros((1,1,3),np.uint8)),scene


@pytest.mark.parametrize('damaged_side',['a','b'])
def test_sharper_source_is_selected_independently_of_input_order(damaged_side):
    reg,scene=pair(damaged_side);a=reg.canvas_a.copy();b=reg.canvas_b.copy()
    result,metrics=fuse(reg,{'mode':'adaptive_detail','guard':True})
    assert metrics['selection']==('b' if damaged_side=='a' else 'a')
    region=np.s_[20:-20,145:255]
    before=np.mean(np.abs(feather_blend(reg)[region].astype(float)-scene[region]))
    after=np.mean(np.abs(result[region].astype(float)-scene[region]))
    assert after<before*.65
    np.testing.assert_array_equal(a,reg.canvas_a);np.testing.assert_array_equal(b,reg.canvas_b)
    only_a=(reg.mask_a>0)&(reg.mask_b==0);only_b=(reg.mask_b>0)&(reg.mask_a==0)
    np.testing.assert_array_equal(result[only_a],a[only_a]);np.testing.assert_array_equal(result[only_b],b[only_b])


def test_already_matching_images_are_kept_exactly():
    reg,_=pair(None)
    result,metrics=fuse(reg,{'mode':'adaptive_detail','guard':True})
    np.testing.assert_array_equal(result,feather_blend(reg))
    assert metrics['kept_aligned_input']


def test_smooth_photometric_difference_does_not_trigger_texture_selection():
    reg, _ = pair(None)
    height, width = reg.canvas_a.shape[:2]
    gradient = np.linspace(.7, 1.3, height, dtype=np.float32)[:, None, None]
    reg.canvas_a = np.clip(reg.canvas_a.astype(np.float32) * gradient, 0, 255).astype(np.uint8)
    fraction, magnitude = low_frequency_photometric_disagreement(
        reg.canvas_a, reg.canvas_b, reg.overlap_mask,
    )
    assert fraction > .9 and magnitude > 2.5
    _, metrics = fuse(reg, {'mode': 'adaptive_detail', 'guard': True, 'clarity_margin': 1.01})
    assert metrics['smooth_photometric_fraction'] > .9
    assert metrics['selection'] == 'balanced; preserve feather'


@pytest.mark.parametrize('noisy_side',['a','b'])
@pytest.mark.parametrize('contrast,noise_std',[(1,14),(.1,2)])
def test_noise_is_not_mistaken_for_sharp_texture(noisy_side,contrast,noise_std):
    reg,scene=pair(None)
    # A smooth, structured surface with independent sensor-like noise in one input.
    scene=cv2.GaussianBlur(scene,(0,0),2.5)
    scene=np.clip(scene.astype(float)*contrast+100*(1-contrast),0,255).astype(np.uint8)
    reg.canvas_a[:,:280]=scene[:,:280];reg.canvas_b[:,120:]=scene[:,120:]
    noisy=reg.canvas_a if noisy_side=='a' else reg.canvas_b
    support=reg.mask_a if noisy_side=='a' else reg.mask_b
    random=np.random.default_rng(14).normal(0,noise_std,noisy.shape)
    noisy[support>0]=np.clip(noisy.astype(float)+random,0,255).astype(np.uint8)[support>0]
    result,metrics=fuse(reg,{'mode':'adaptive_detail','guard':True})
    assert metrics['noise_guard_applied']
    assert metrics['selection']==('b' if noisy_side=='a' else 'a')
    region=np.s_[20:-20,145:255]
    assert np.abs(result[region].astype(float)-scene[region]).mean()<np.abs(feather_blend(reg)[region].astype(float)-scene[region]).mean()*.7


def test_pipeline_can_export_clear_result_without_ai(monkeypatch):
    reg,_=pair('b')
    monkeypatch.setattr('xiyuan_mvp.pipeline.register_images',lambda *a,**k:reg)
    def forbidden(*a,**k):raise AssertionError('Clear fusion does not require AI')
    monkeypatch.setattr('xiyuan_mvp.neural_alignment.repair_alignment',forbidden)
    monkeypatch.setattr('xiyuan_mvp.inpainting.DiffusersInpainter.generate',forbidden)
    config=load_config();config['clear_fusion']['enabled']=True
    result=StitchPipeline(config).run(reg.image_a,reg.image_b)
    assert result.metrics['use_ai'] is False
    assert result.metrics['clear_fusion_enabled'] is True
    assert result.metrics['clear_selection']=='a'
    assert not np.array_equal(result.final_image,result.traditional_image)
    np.testing.assert_array_equal(result.final_image,result.repair_base_image)
