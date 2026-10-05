"""Learned local alignment for repair, using only the two input photographs."""
from dataclasses import replace
from functools import lru_cache
import time
import cv2
import numpy as np

from .blending import feather_blend
from .errors import InpaintingUnavailableError


@lru_cache(maxsize=2)
def _raft(device, large=False):
    import torch
    if device == 'cpu':
        torch.set_num_threads(4)
    from torchvision.models.optical_flow import raft_small, raft_large, Raft_Small_Weights, Raft_Large_Weights
    weights=Raft_Large_Weights.C_T_SKHT_V2 if large else Raft_Small_Weights.C_T_V2
    model=(raft_large if large else raft_small)(weights=weights,progress=False).eval().to(device)
    return model,weights.url


def _exposure(a,b,mask):
    region=cv2.erode(mask,np.ones((9,9),np.uint8))>0
    if region.sum()<128:region=mask>0
    corrected=b.astype(np.float32)
    for ch in range(3):
        av=a[:,:,ch][region].astype(float);bv=b[:,:,ch][region].astype(float)
        aq=np.percentile(av,[10,50,90]);bq=np.percentile(bv,[10,50,90])
        gain=np.clip((aq[2]-aq[0])/max(bq[2]-bq[0],8),.7,1.4)
        shift=np.clip(aq[1]-gain*bq[1],-40,40)
        corrected[:,:,ch]=corrected[:,:,ch]*gain+shift
    return np.clip(np.rint(corrected),0,255).astype(np.uint8)


def _local_illumination(a,b,mask):
    """Smooth input-only affine field; correct varying light without repainting texture."""
    valid=(mask>0).astype(np.float32)
    sigma=24.0
    weight=cv2.GaussianBlur(valid,(0,0),sigma)
    safe=np.maximum(weight,1e-4)
    output=b.astype(np.float32).copy()
    for ch in range(3):
        ac=a[:,:,ch].astype(np.float32);bc=b[:,:,ch].astype(np.float32)
        def mean(value):return cv2.GaussianBlur(value*valid,(0,0),sigma)/safe
        ma,mb=mean(ac),mean(bc)
        variance=np.maximum(mean(bc*bc)-mb*mb,0)
        covariance=mean(ac*bc)-ma*mb
        # Ridge toward unit gain avoids amplifying noise in near-flat regions.
        gain=np.clip((covariance+16)/(variance+16),.75,1.3)
        shift=np.clip(ma-gain*mb,-40,40)
        corrected=bc*gain+shift
        output[:,:,ch]=np.where(weight>.03,corrected,bc)
    return np.clip(np.rint(output),0,255).astype(np.uint8)


def repair_alignment(registration,base,mask,config,*,method='raft'):
    if config.get('preserve_detail',False) and method=='raft_large':
        from .structural_repair import repair
        return repair(registration,base,mask,config)
    started=time.perf_counter()
    a=registration.canvas_a
    b=registration.canvas_b.copy()
    overlap=registration.overlap_mask
    if config.get('exposure',True):b=_exposure(a,b,overlap)
    b[registration.mask_b==0]=0
    metrics={'repair_engine':method,'exposure_enabled':config.get('exposure',True)}
    x,y,w,h=cv2.boundingRect(overlap)
    ac=a[y:y+h,x:x+w];bc=b[y:y+h,x:x+w]
    valid=overlap[y:y+h,x:x+w]>0
    before=np.mean(np.abs(ac.astype(float)-bc.astype(float)),axis=2)
    metrics['overlap_mae_before']=float(before[valid].mean())
    if method!='exposure_only':
        scale=min(1,float(config.get('max_size',512))/max(h,w))
        size=(max(8,int(w*scale)//8*8),max(8,int(h*scale)//8*8))
        aa=cv2.resize(ac,size);bb=cv2.resize(bc,size)
        if method in ('raft','raft_large','raft_refined'):
            import torch
            device=config.get('device','cpu')
            if device=='cuda' and not torch.cuda.is_available():raise InpaintingUnavailableError('RAFT requested CUDA but it is unavailable')
            model,url=_raft(device, method in ('raft_large','raft_refined'))
            if device=='cuda':torch.cuda.reset_peak_memory_stats()
            def tensor(im):
                ih,iw=im.shape[:2]
                padded=cv2.copyMakeBorder(im,0,max(0,128-ih),0,max(0,128-iw),cv2.BORDER_REFLECT_101)
                return torch.from_numpy(np.ascontiguousarray(padded[:,:,::-1].transpose(2,0,1))).float().unsqueeze(0).to(device)/127.5-1
            ta,tb=tensor(aa),tensor(bb)
            t=time.perf_counter()
            with torch.inference_mode():
                forward=model(ta,tb,num_flow_updates=int(config.get('iterations',12)))[-1][0,:,:size[1],:size[0]].permute(1,2,0).cpu().numpy()
                reverse=model(tb,ta,num_flow_updates=int(config.get('iterations',12)))[-1][0,:,:size[1],:size[0]].permute(1,2,0).cpu().numpy()
            metrics.update(model='torchvision/raft_large/C_T_SKHT_V2' if method in ('raft_large','raft_refined') else 'torchvision/raft_small/C_T_V2',model_url=url,
                           model_hash_prefix=url.rsplit('-',1)[-1].split('.')[0],
                           device=device,flow_inference_seconds=time.perf_counter()-t)
            if device=='cuda':metrics['peak_vram_mb']=torch.cuda.max_memory_allocated()/1024**2
            if method=='raft_refined':
                ga=cv2.cvtColor(aa,cv2.COLOR_BGR2GRAY);gb=cv2.cvtColor(bb,cv2.COLOR_BGR2GRAY)
                dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
                forward=dis.calc(ga,gb,np.ascontiguousarray(forward))
                reverse=dis.calc(gb,ga,np.ascontiguousarray(reverse))
        elif method=='dis':
            ga=cv2.cvtColor(aa,cv2.COLOR_BGR2GRAY);gb=cv2.cvtColor(bb,cv2.COLOR_BGR2GRAY)
            dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
            forward=dis.calc(ga,gb,None);reverse=dis.calc(gb,ga,None)
        else:raise ValueError('Unknown alignment method')
        yy,xx=np.indices(forward.shape[:2],dtype=np.float32)
        inverse=cv2.remap(reverse,xx+forward[:,:,0],yy+forward[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        consistency=np.linalg.norm(forward+inverse,axis=2)
        magnitude=np.linalg.norm(forward,axis=2)
        confident=(consistency<float(config.get('consistency',1.5)))&(magnitude<float(config.get('max_flow',24))*scale)
        flow=cv2.resize(forward,(w,h));flow[:,:,0]*=w/size[0];flow[:,:,1]*=h/size[1]
        conf=cv2.resize(confident.astype(np.float32),(w,h))*valid
        # Padding is necessary: distanceTransform of an all-valid rectangular
        # overlap otherwise has no zero boundary and returns a huge distance.
        distance=cv2.distanceTransform(np.pad(valid.astype(np.uint8),1),(cv2.DIST_L2),5)[1:-1,1:-1]
        weight=cv2.GaussianBlur(conf,(9,9),0)*np.clip(distance/12,0,1)*valid
        flow*=weight[:,:,None]
        yy,xx=np.indices((h,w),dtype=np.float32)
        corrected=cv2.remap(bc,xx+flow[:,:,0],yy+flow[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
        after=np.mean(np.abs(ac.astype(float)-corrected.astype(float)),axis=2)
        accepted=after[valid].mean()<before[valid].mean()*.99
        if accepted:b[y:y+h,x:x+w]=corrected
        metrics.update(flow_accepted=bool(accepted),flow_confident_fraction=float(conf[valid].mean()),
                       overlap_mae_after=float(after[valid].mean()))
    if config.get('local_illumination',False):
        b=_local_illumination(a,b,overlap)
        b[registration.mask_b==0]=0
        metrics['local_illumination']=True
    candidate=feather_blend(replace(registration,canvas_b=b))
    alpha=(mask.soft_mask.astype(float)/255)[:,:,None]
    result=np.clip(np.rint(base.astype(float)*(1-alpha)+candidate.astype(float)*alpha),0,255).astype(np.uint8)
    result[mask.soft_mask==0]=base[mask.soft_mask==0]
    metrics['repair_seconds']=time.perf_counter()-started
    return result,metrics
