import cv2
import numpy as np

from xiyuan_mvp.blending import poisson_blend
from xiyuan_mvp.types import RegistrationResult


def test_poisson_keeps_cloned_gradients_despite_opencv_mutating_its_mask():
    a=np.full((120,160,3),100,np.uint8)
    b=np.full_like(a,120)
    cv2.circle(b,(80,60),12,(245,245,245),-1)
    mask_a=np.full(a.shape[:2],255,np.uint8)
    mask_b=np.zeros_like(mask_a);mask_b[10:110,40:120]=255
    overlap=mask_b.copy()
    reg=RegistrationResult(a,b,a.copy(),b.copy(),mask_a,mask_b,overlap,np.eye(3),np.eye(3),a.copy())
    result=poisson_blend(reg)
    # Poisson should transfer the source's bright circle gradient. Restoring
    # pixels based on OpenCV's internally-eroded mask used to erase this result.
    assert result[60,80].mean()>180
    np.testing.assert_array_equal(result[overlap==0],a[overlap==0])
    np.testing.assert_array_equal(reg.overlap_mask,overlap)
