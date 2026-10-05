"""Image-only photometric and local geometric preparation for constrained inpainting.

Reference panoramas and benchmark labels are never inputs to this module.
"""
from __future__ import annotations

from dataclasses import replace
import cv2
import numpy as np

from .seam_mask import _minimum_cost_seam, _center_seam
from .types import RegistrationResult


def prepare_structural_base(registration: RegistrationResult, config: dict):
    a=registration.canvas_a
    b=registration.canvas_b.astype(np.float32)
    overlap=registration.overlap_mask
    region=cv2.erode(overlap,np.ones((9,9),np.uint8))>0
    if np.count_nonzero(region)<128:
        region=overlap>0
    metrics={"refinement_enabled":True}
    if config.get("exposure",True):
        gains,shifts=[],[]
        for channel in range(3):
            av=a[:,:,channel][region].astype(np.float32)
            bv=b[:,:,channel][region]
            aq=np.percentile(av,[10,50,90]);bq=np.percentile(bv,[10,50,90])
            gain=float(np.clip((aq[2]-aq[0])/max(bq[2]-bq[0],8),0.70,1.40))
            shift=float(np.clip(aq[1]-gain*bq[1],-40,40))
            b[:,:,channel]=np.clip(b[:,:,channel]*gain+shift,0,255)
            gains.append(gain);shifts.append(shift)
        metrics.update(exposure_gains_bgr=gains,exposure_shifts_bgr=shifts)
    b=b.astype(np.uint8)
    b[registration.mask_b==0]=0
    metrics["overlap_mae_before_flow"]=float(np.abs(a.astype(float)-b.astype(float))[region].mean())
    metrics["local_flow_applied"]=False
    if config.get("local_flow",True) and metrics["overlap_mae_before_flow"]>float(config.get("flow_min_mae",2.0)):
        x,y,w,h=cv2.boundingRect(overlap)
        ac=a[y:y+h,x:x+w]
        bc=b[y:y+h,x:x+w]
        valid=overlap[y:y+h,x:x+w]>0
        ga=cv2.cvtColor(ac,cv2.COLOR_BGR2GRAY)
        gb=cv2.cvtColor(bc,cv2.COLOR_BGR2GRAY)
        ga[~valid]=0;gb[~valid]=0
        scale=min(1.0,800/max(h,w))
        size=(max(12,int(w*scale)),max(12,int(h*scale)))
        sa=cv2.resize(ga,size);sb=cv2.resize(gb,size)
        dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        forward=dis.calc(sa,sb,None)
        reverse=dis.calc(sb,sa,None)
        yy,xx=np.indices(sa.shape,dtype=np.float32)
        projected_reverse=cv2.remap(reverse,xx+forward[:,:,0],yy+forward[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        consistent=np.linalg.norm(forward+projected_reverse,axis=2)<1.5
        magnitude=np.linalg.norm(forward,axis=2)
        consistent &= magnitude<float(config.get("max_flow_pixels",16))*scale
        confidence=cv2.resize(consistent.astype(np.float32),(w,h),interpolation=cv2.INTER_LINEAR)
        confidence*=valid
        distance=cv2.distanceTransform(valid.astype(np.uint8),cv2.DIST_L2,5)
        weight=confidence*np.clip(distance/24,0,1)
        weight=cv2.GaussianBlur(weight,(17,17),0)*valid
        flow=cv2.resize(forward,(w,h),interpolation=cv2.INTER_LINEAR)
        flow[:,:,0]*=w/size[0];flow[:,:,1]*=h/size[1]
        flow*=weight[:,:,None]
        yy,xx=np.indices((h,w),dtype=np.float32)
        corrected=cv2.remap(bc,xx+flow[:,:,0],yy+flow[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
        before=np.mean(np.abs(ac.astype(float)-bc.astype(float)),axis=2)
        after=np.mean(np.abs(ac.astype(float)-corrected.astype(float)),axis=2)
        # Accept a field only when image agreement improves over the common valid area.
        if float(after[valid].mean())<float(before[valid].mean())*0.98:
            b[y:y+h,x:x+w]=corrected
            metrics["local_flow_applied"]=True
        metrics["flow_consistent_fraction"]=float(confidence[valid].mean())
    metrics["overlap_mae_after_flow"]=float(np.abs(a.astype(float)-b.astype(float))[region].mean())
    refined=replace(registration,canvas_b=b)
    line=_minimum_cost_seam(refined)
    if line is None:
        line=_center_seam(overlap)
    ys,xs=np.where(overlap>0)
    vertical=(xs.max()-xs.min())<=(ys.max()-ys.min())
    axes=(1,0) if vertical else (0,1)
    ca=np.mean(np.argwhere(registration.mask_a>0),axis=0)
    cb=np.mean(np.argwhere(registration.mask_b>0),axis=0)
    first_is_lower=ca[axes[0]]<=cb[axes[0]]
    alpha=(registration.mask_a>0).astype(np.float32)
    for index in range(line.shape[0] if vertical else line.shape[1]):
        points=np.flatnonzero(line[index] if vertical else line[:,index])
        if not points.size:
            continue
        seam=float(np.median(points))
        coordinates=np.arange(line.shape[1] if vertical else line.shape[0])
        values=np.clip((seam-coordinates)/float(config.get("seam_softness",8))+0.5,0,1)
        if not first_is_lower:
            values=1-values
        if vertical:
            eligible=overlap[index]>0
            alpha[index,eligible]=values[eligible]
        else:
            eligible=overlap[:,index]>0
            alpha[eligible,index]=values[eligible]
    alpha[(registration.mask_a==0)&(registration.mask_b>0)]=0
    base=np.clip(a.astype(float)*alpha[:,:,None]+b.astype(float)*(1-alpha[:,:,None]),0,255).astype(np.uint8)
    base[(registration.mask_a|registration.mask_b)==0]=0
    return base,refined,metrics
