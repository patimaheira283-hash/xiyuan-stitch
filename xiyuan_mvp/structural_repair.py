"""Symmetric learned alignment with input-only acceptance and detail preservation."""
from dataclasses import replace
import time
import cv2
import numpy as np

from .clear_fusion import fuse
from .neural_alignment import _raft

CLEAR = {'mode': 'adaptive_detail', 'detail_scale': 9, 'guard': True}
PREDICTION_REVISION = 'symmetric-exposure-v1'


def clear_settings(config):
    """Use the same fusion settings as the input base; retain legacy API defaults."""
    return {**CLEAR, **config.get('clear_fusion', {})}


def normalize_pair(a,b,mask):
    """Normalize both matching inputs to the same robust photometric midpoint."""
    region=cv2.erode(mask,np.ones((9,9),np.uint8))>0
    if region.sum()<128:region=mask>0
    out=[a.astype(np.float32),b.astype(np.float32)]
    for ch in range(3):
        quantiles=[np.percentile(im[:,:,ch][region],[10,50,90]) for im in (a,b)]
        target=(quantiles[0]+quantiles[1])*.5
        for im,q in zip(out,quantiles):
            gain=np.clip((target[2]-target[0])/max(q[2]-q[0],8),.7,1.4)
            shift=np.clip(target[1]-gain*q[1],-40,40)
            im[:,:,ch]=im[:,:,ch]*gain+shift
    return [np.clip(np.rint(im),0,255).astype(np.uint8) for im in out]


def predict_fields(registration, config):
    import torch
    started = time.perf_counter()
    x, y, w, h = cv2.boundingRect(registration.overlap_mask)
    if min(w, h) < 16:
        raise ValueError('重叠区域过小，无法估计局部位移。')
    a = registration.canvas_a[y:y+h, x:x+w]
    b = registration.canvas_b[y:y+h, x:x+w]
    valid = registration.overlap_mask[y:y+h, x:x+w]
    scale = min(1., float(config.get('max_size', 512))/max(w, h))
    size = (max(8, int(w*scale)//8*8), max(8, int(h*scale)//8*8))
    # Exposure normalization is only used to estimate flow, never to recolor output.
    if config.get('exposure', True):a,b=normalize_pair(a,b,valid)
    device = config.get('device', 'cpu')
    model, url = _raft(device, True)
    def tensor(im):
        im = cv2.resize(im, size)
        im = cv2.copyMakeBorder(im, 0, max(0,128-im.shape[0]), 0,
                               max(0,128-im.shape[1]), cv2.BORDER_REFLECT_101)
        return torch.from_numpy(np.ascontiguousarray(im[:,:,::-1].transpose(2,0,1))).float()[None].to(device)/127.5-1
    aa, bb = tensor(a), tensor(b)
    if device == 'cuda':
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    inference = time.perf_counter()
    fields = []
    with torch.inference_mode():
        for first, second in [(aa,bb),(bb,aa)]:
            raw = model(first, second, num_flow_updates=int(config.get('iterations',12)))[-1]
            raw = raw[0,:,:size[1],:size[0]].permute(1,2,0).cpu().numpy()
            field = cv2.resize(raw, (w,h))
            field[:,:,0] *= w/size[0]; field[:,:,1] *= h/size[1]
            fields.append(field)
    metrics = {'model':'torchvision/raft_large/C_T_SKHT_V2', 'model_url':url,
               'prediction_revision':PREDICTION_REVISION,'model_executed':True,
               'device':device, 'flow_inference_seconds':time.perf_counter()-inference,
               'prediction_seconds':time.perf_counter()-started}
    if device == 'cuda':
        metrics['peak_vram_mb'] = torch.cuda.max_memory_allocated()/1024**2
        metrics['peak_reserved_vram_mb'] = torch.cuda.max_memory_reserved()/1024**2
    return fields, metrics


def compose_fields(registration, base, mask, fields, config, *, strategy='clarity'):
    started = time.perf_counter()
    clear_config = clear_settings(config)
    clear, original = fuse(registration, clear_config)
    metrics = {'repair_engine':'raft_large', 'repair_revision':'clarity-guided-raft-v1',
               'repair_clear_config':clear_config,
               'source_selection':original.get('selection','already aligned')}
    region=mask.soft_mask>0
    if config.get('only_balanced', True) and original.get('selection') in ('a','b') and np.array_equal(base[region],clear[region]):
        metrics.update(repair_applied=False, repair_reason='clear_source_already_selected')
        return base.copy(), metrics
    x,y,w,h = cv2.boundingRect(registration.overlap_mask)
    valid = registration.overlap_mask[y:y+h,x:x+w]>0
    yy,xx = np.indices((h,w),dtype=np.float32)
    distance = cv2.distanceTransform(np.pad(valid.astype(np.uint8),1),cv2.DIST_L2,5)[1:-1,1:-1]
    trusted = []
    for field, reverse in [(fields[0],fields[1]),(fields[1],fields[0])]:
        sx,sy = xx+field[:,:,0], yy+field[:,:,1]
        inverse = cv2.remap(reverse,sx,sy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        support = cv2.remap(valid.astype(np.uint8),sx,sy,cv2.INTER_NEAREST)>0
        good = (np.linalg.norm(field+inverse,axis=2)<float(config.get('consistency',1.5)))
        good &= np.linalg.norm(field,axis=2)<float(config.get('max_flow',24))
        good &= support & valid & (sx>=0) & (sy>=0) & (sx<w-1) & (sy<h-1)
        confidence = cv2.GaussianBlur(good.astype(np.float32),(9,9),0)*np.clip(distance/16,0,1)*valid
        trusted.append(field*confidence[:,:,None])
    a,b = registration.canvas_a.copy(),registration.canvas_b.copy()
    ac,bc = a[y:y+h,x:x+w].copy(),b[y:y+h,x:x+w].copy()
    # For two similarly sharp sources, move both toward a common geometry.
    # A slight clarity advantage biases the shared geometry toward that input,
    # smoothly, instead of always treating the first photograph as ground truth.
    ratio=(original.get('clarity_a',1)+.01)/(original.get('clarity_b',1)+.01)
    margin = float(clear_config.get('clarity_margin', 1.1))
    anchor_weight=float(np.clip(.5+np.log(max(ratio,1e-6))/(2*np.log(margin)),0,1)) if strategy=='clarity' else (.5 if strategy=='midpoint' else 1.)
    metrics['anchor_a_weight']=anchor_weight
    fa,fb = trusted[1]*(1-anchor_weight),trusted[0]*anchor_weight
    aa = cv2.remap(ac,xx+fa[:,:,0],yy+fa[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
    bb = cv2.remap(bc,xx+fb[:,:,0],yy+fb[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
    for output,original_image,support,field in [(aa,ac,registration.mask_a,fa),(bb,bc,registration.mask_b,fb)]:
        covered=cv2.remap(support[y:y+h,x:x+w],xx+field[:,:,0],yy+field[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        output[covered<255]=original_image[covered<255]
    def residual(im):
        g = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(np.float32)
        return g-cv2.GaussianBlur(g,(0,0),3)
    before = np.abs(residual(ac)-residual(bc))
    after = np.abs(residual(aa)-residual(bb))
    region = valid & (mask.soft_mask[y:y+h,x:x+w]>0) & (distance>8)
    if not region.any():
        metrics.update(repair_applied=False, repair_reason='empty_reliable_region')
        return base.copy(),metrics
    old,new = float(before[region].mean()),float(after[region].mean())
    metrics.update(structural_error_before=old,structural_error_after=new)
    if new >= old*.95 or old<.5:
        metrics.update(repair_applied=False,repair_reason='insufficient_alignment_gain')
        return base.copy(),metrics
    a[y:y+h,x:x+w]=aa;b[y:y+h,x:x+w]=bb
    candidate, clear_metrics = fuse(replace(registration,canvas_a=a,canvas_b=b),clear_config)
    alpha = mask.soft_mask.astype(np.float32)/255
    out = np.clip(np.rint(base.astype(np.float32)*(1-alpha[:,:,None])+candidate.astype(np.float32)*alpha[:,:,None]),0,255).astype(np.uint8)
    out[mask.soft_mask==0]=base[mask.soft_mask==0]
    metrics.update(repair_applied=True,repair_reason='accepted',composition_selection=clear_metrics.get('selection'),
                   composition_seconds=time.perf_counter()-started)
    return out,metrics


def repair(registration,base,mask,config):
    started=time.perf_counter()
    clear_config = clear_settings(config)
    clear,original=fuse(registration,clear_config)
    region=mask.soft_mask>0
    reason=None
    if original.get('kept_aligned_input'):reason='already_aligned'
    elif config.get('only_balanced',True) and original.get('selection') in ('a','b') and np.array_equal(base[region],clear[region]):
        reason='clear_source_already_selected'
    if reason:
        return base.copy(),{'repair_engine':'raft_large','repair_revision':'clarity-guided-raft-v1',
            'repair_clear_config':clear_config,
            'repair_applied':False,'model_executed':False,'repair_reason':reason,
            'repair_seconds':time.perf_counter()-started}
    fields,metrics=predict_fields(registration,config)
    result,composition=compose_fields(registration,base,mask,fields,config)
    return result,{**metrics,**composition,'repair_seconds':time.perf_counter()-started}
