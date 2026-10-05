from dataclasses import replace
import cv2
import numpy as np

from xiyuan_mvp.clear_fusion import fuse
from xiyuan_mvp.seam_mask import mask_from_binary
from xiyuan_mvp.structural_repair import CLEAR, compose_fields, normalize_pair, repair
from tests.test_clear_fusion import pair


def shifted_pair():
    reg,_=pair(None)
    h,w=reg.canvas_b.shape[:2]
    yy,xx=np.indices((h,w),dtype=np.float32)
    reg.canvas_b[:]=cv2.remap(reg.canvas_b,xx+4,yy,cv2.INTER_LINEAR)
    reg.canvas_b[reg.mask_b==0]=0
    return reg


def test_photometric_normalization_is_symmetric():
    reg,_=pair(None)
    a=reg.canvas_a;b=np.clip(reg.canvas_b.astype(float)*.75,0,255).astype(np.uint8)
    ab=normalize_pair(a,b,reg.overlap_mask);ba=normalize_pair(b,a,reg.overlap_mask)
    np.testing.assert_array_equal(ab[0],ba[1]);np.testing.assert_array_equal(ab[1],ba[0])


def test_known_motion_reduces_disagreement_and_preserves_mask_exterior():
    reg=shifted_pair();base,_=fuse(reg,CLEAR)
    painted=reg.overlap_mask.copy();painted[:45]=0;painted[135:]=0
    mask=mask_from_binary(painted,reg.overlap_mask,8)
    x,y,w,h=cv2.boundingRect(reg.overlap_mask)
    forward=np.zeros((h,w,2),np.float32);forward[:,:,0]=-4
    out,metrics=compose_fields(reg,base,mask,[forward,-forward],{'only_balanced':False})
    assert metrics['repair_applied']
    assert metrics['structural_error_after']<metrics['structural_error_before']*.4
    np.testing.assert_array_equal(out[mask.soft_mask==0],base[mask.soft_mask==0])
    swapped=replace(reg,canvas_a=reg.canvas_b,canvas_b=reg.canvas_a,mask_a=reg.mask_b,mask_b=reg.mask_a)
    reverse,rm=compose_fields(swapped,base,mask,[-forward,forward],{'only_balanced':False})
    # Swapping sources can change the existing float feather rounding by one
    # code value; the geometric correction itself must remain symmetric.
    assert np.abs(out.astype(int)-reverse.astype(int)).max()<=1
    assert np.abs(out.astype(int)-reverse.astype(int)).mean()<.01


def test_rejected_motion_keeps_the_existing_base_exactly():
    reg=shifted_pair();base,_=fuse(reg,CLEAR)
    mask=mask_from_binary(reg.overlap_mask,reg.overlap_mask,8)
    x,y,w,h=cv2.boundingRect(reg.overlap_mask)
    out,metrics=compose_fields(reg,base,mask,[np.zeros((h,w,2),np.float32)]*2,{'only_balanced':False})
    assert not metrics['repair_applied']
    np.testing.assert_array_equal(out,base)


def test_pipeline_and_raft_use_same_nondefault_clear_configuration(monkeypatch):
    from xiyuan_mvp.config import load_config
    from xiyuan_mvp.pipeline import StitchPipeline
    reg, _ = pair('b')
    monkeypatch.setattr('xiyuan_mvp.pipeline.register_images', lambda *a, **k: reg)
    def forbidden(*a, **k):
        raise AssertionError('A selected clear source must not be warped using different fusion parameters')
    monkeypatch.setattr('xiyuan_mvp.structural_repair.predict_fields', forbidden)
    cfg = load_config()
    cfg['inpainting']['engine'] = 'neural_alignment'
    cfg['clear_fusion'].update(detail_scale=3, clarity_margin=1.05, transition_width=24)
    pipe = StitchPipeline(cfg)
    base = pipe.run(reg.image_a, reg.image_b)
    result = pipe.run(reg.image_a, reg.image_b, prepared=base, use_ai=True)
    assert not result.metrics['model_executed']
    assert result.metrics['repair_clear_config']['detail_scale'] == 3
    np.testing.assert_array_equal(result.final_image, base.final_image)


def test_clear_source_protection_skips_unnecessary_network_inference(monkeypatch):
    reg,_=pair('b');base,_=fuse(reg,CLEAR)
    mask=mask_from_binary(reg.overlap_mask,reg.overlap_mask,8)
    def forbidden(*a,**kw):raise AssertionError('No model needed for an already selected clear source')
    monkeypatch.setattr('xiyuan_mvp.structural_repair.predict_fields',forbidden)
    out,metrics=repair(reg,base,mask,{})
    assert not metrics['model_executed']
    np.testing.assert_array_equal(out,base)
