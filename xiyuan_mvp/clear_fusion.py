"""Content-preserving fusion experiments. No reference images or case labels enter here."""
from dataclasses import replace
import time
import cv2
import numpy as np
from .blending import feather_blend


def graphcut_masks(registration, a, b, max_size=720):
    h,w=a.shape[:2]
    scale=min(1,max_size/max(h,w));size=(max(16,round(w*scale)),max(16,round(h*scale)))
    inputs=[cv2.resize(i,size).astype(np.float32) for i in (a,b)]
    masks=[cv2.UMat(cv2.resize(m,size,interpolation=cv2.INTER_NEAREST))
           for m in (registration.mask_a,registration.mask_b)]
    finder=cv2.detail_GraphCutSeamFinder('COST_COLOR_GRAD')
    found=finder.find(inputs,[(0,0),(0,0)],masks)
    if found is None:found=masks
    return [cv2.bitwise_and(cv2.resize(m.get() if hasattr(m,'get') else m,(w,h),
                interpolation=cv2.INTER_NEAREST),valid)
            for m,valid in zip(found,(registration.mask_a,registration.mask_b))]


def multiband(a,b,masks,bands=3):
    h,w=a.shape[:2]
    blender=cv2.detail_MultiBandBlender(0,bands)
    blender.prepare((0,0,w,h))
    for im,mask in zip((a,b),masks):blender.feed(im.astype(np.int16),mask,(0,0))
    result,_=blender.blend(None,None)
    return np.clip(result,0,255).astype(np.uint8)


def dense_align(registration):
    """Bounded full-resolution DIS, with a bidirectional reliability gate."""
    a,b=registration.canvas_a,registration.canvas_b.copy()
    x,y,w,h=cv2.boundingRect(registration.overlap_mask)
    ac=a[y:y+h,x:x+w];bc=b[y:y+h,x:x+w]
    size=(max(16,round(w*min(1,900/max(h,w)))),max(16,round(h*min(1,900/max(h,w)))))
    def gray(im):
        g=cv2.cvtColor(cv2.resize(im,size),cv2.COLOR_BGR2GRAY)
        return cv2.createCLAHE(clipLimit=2,tileGridSize=(8,8)).apply(g)
    aa,bb=gray(ac),gray(bc)
    dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    f=dis.calc(aa,bb,None);rev=dis.calc(bb,aa,None)
    yy,xx=np.indices(aa.shape,dtype=np.float32)
    inverse=cv2.remap(rev,xx+f[:,:,0],yy+f[:,:,1],cv2.INTER_LINEAR)
    confidence=(np.linalg.norm(f+inverse,axis=2)<1.5)&(np.linalg.norm(f,axis=2)<24*size[0]/w)
    field=cv2.resize(f,(w,h));field[:,:,0]*=w/size[0];field[:,:,1]*=h/size[1]
    valid=(registration.overlap_mask[y:y+h,x:x+w]>0).astype(np.uint8)
    dist=cv2.distanceTransform(np.pad(valid,1),cv2.DIST_L2,5)[1:-1,1:-1]
    weight=cv2.GaussianBlur(cv2.resize(confidence.astype(np.float32),(w,h)),(7,7),0)*np.clip(dist/16,0,1)
    field*=weight[:,:,None]
    yy,xx=np.indices((h,w),dtype=np.float32)
    warped=cv2.remap(bc,xx+field[:,:,0],yy+field[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
    # Compare local high-pass structure, so changing illumination does not force a warp.
    def residual(p):
        return p.astype(np.float32)-cv2.GaussianBlur(p.astype(np.float32),(0,0),3)
    before=np.abs(residual(ac)-residual(bc)).mean(axis=2)
    after=np.abs(residual(ac)-residual(warped)).mean(axis=2)
    region=valid>0
    accepted=float(after[region].mean())<float(before[region].mean())*.97
    if accepted:b[y:y+h,x:x+w]=warped
    return b,{'flow_applied':accepted,'structural_error_before':float(before[region].mean()),
              'structural_error_after':float(after[region].mean())}


def structural_agreement(a,b,overlap):
    region=cv2.erode(overlap,np.ones((9,9),np.uint8))>0
    if region.sum()<64:region=overlap>0
    def detail(im):
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(np.float32)
        return gray-cv2.GaussianBlur(gray,(0,0),3)
    da,db=detail(a)[region],detail(b)[region]
    return float(np.dot(da,db)/max(np.linalg.norm(da)*np.linalg.norm(db),1e-6))


def source_clarity(image,overlap,*,with_energy=False):
    """Contrast-normalized multiscale detail ratio; suppress pixel noise before scoring."""
    g=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.float32)
    g0=cv2.GaussianBlur(g,(0,0),.6);g1=cv2.GaussianBlur(g,(0,0),1.6);g2=cv2.GaussianBlur(g,(0,0),3.2)
    region=cv2.erode(overlap,np.ones((15,15),np.uint8))>0
    if region.sum()<64:region=overlap>0
    fine=float(np.mean((g0-g1)[region]**2));coarse=float(np.mean((g1-g2)[region]**2))
    ratio=fine/max(coarse,.01)
    return (ratio,fine) if with_energy else ratio


def noise_floor(image,overlap):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.float32)
    kernel=np.array([[1,-2,1],[-2,4,-2],[1,-2,1]],np.float32)
    response=cv2.filter2D(gray,-1,kernel)
    valid=cv2.erode(overlap,np.ones((15,15),np.uint8))>0
    if valid.sum()<64:valid=overlap>0
    return float(np.median(np.abs(response[valid]))/(6*.67448975))


def low_frequency_photometric_disagreement(a, b, overlap):
    """Estimate whether the inter-image difference is mostly smooth lighting.

    It is deliberately input-only. A high value means that selecting the
    apparently sharper source could be selecting a lighting gradient rather
    than real texture detail.
    """
    x, y, w, h = cv2.boundingRect(overlap)
    if not w or not h:
        return 0.0, 0.0
    aa = cv2.cvtColor(a[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY).astype(np.float32)
    bb = cv2.cvtColor(b[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY).astype(np.float32)
    valid = overlap[y:y+h, x:x+w] > 0
    if not np.any(valid):
        return 0.0, 0.0
    difference = aa - bb
    smooth = cv2.GaussianBlur(difference, (0, 0), 5.0)
    values = difference[valid]
    smooth_values = smooth[valid]
    total = float(np.mean(np.abs(values)))
    fraction = float(np.mean(np.abs(smooth_values)) / max(total, 1e-6))
    return fraction, total


def blur_explanation(sharp,soft,overlap):
    """Does smoothing the apparent sharper input explain the other image?

    White-noise-like legitimate textures also have a high noise-floor estimate.
    Keep those when the second input is demonstrably a blurred observation.
    """
    a=cv2.cvtColor(sharp,cv2.COLOR_BGR2GRAY).astype(np.float32)
    b=cv2.cvtColor(soft,cv2.COLOR_BGR2GRAY).astype(np.float32)
    valid=cv2.erode(overlap,np.ones((21,21),np.uint8))>0
    if valid.sum()<64:valid=overlap>0
    target=b[valid].astype(np.float64);target-=target.mean()
    def error(sigma):
        value=cv2.GaussianBlur(a,(0,0),max(sigma,.01))[valid].astype(np.float64)
        value-=value.mean()
        gain=np.clip(np.dot(value,target)/max(np.dot(value,value),1e-6),.7,1.4)
        return float(np.sqrt(np.mean((target-gain*value)**2)))
    lo,hi=0.,2.5
    for _ in range(9):
        m1=lo+(hi-lo)/3;m2=hi-(hi-lo)/3
        if error(m1)<error(m2):hi=m2
        else:lo=m1
    return error((lo+hi)/2)


def detail_compose(registration,a,b,masks,scale=5,alpha=None):
    """Keep smooth scene illumination, take texture from a single source across a narrow seam."""
    # Keep subpixel color values until the final rounding; otherwise truncating
    # an intermediate feather image adds a visible dark bias in smooth regions.
    base=feather_blend(registration,float_output=True)
    fa,fb=a.astype(np.float32),b.astype(np.float32)
    if alpha is None:
        valid_a=(masks[0]>0).astype(np.float32);valid_b=(masks[1]>0).astype(np.float32)
        alpha=valid_a/np.maximum(valid_a+valid_b,1)
        alpha=cv2.GaussianBlur(alpha,(0,0),1.2)
    # All high-pass filtering uses normalized valid support to prevent black borders
    # outside a warped photograph from becoming artificial bright/dark edges.
    def low(image,mask):
        valid=(mask>0).astype(np.float32)
        weight=np.maximum(cv2.GaussianBlur(valid,(0,0),scale),1e-5)
        return cv2.GaussianBlur(image*valid[:,:,None],(0,0),scale)/weight[:,:,None]
    la=low(fa,registration.mask_a);lb=low(fb,registration.mask_b)
    lo=low(base,cv2.bitwise_or(registration.mask_a,registration.mask_b))
    texture=(fa-la)*alpha[:,:,None]+(fb-lb)*(1-alpha[:,:,None])
    return np.clip(np.rint(lo+texture),0,255).astype(np.uint8)


def fuse(registration,config=None,*,masks=None):
    cfg=config or {};started=time.perf_counter()
    a,b=registration.canvas_a,registration.canvas_b
    metrics={'fusion_method':cfg.get('mode','graphcut'),'algorithm_revision':'adaptive-noise-photometric-v4'}
    agreement=structural_agreement(a,b,registration.overlap_mask)
    metrics['structural_agreement']=agreement
    if cfg.get('guard',False) and agreement>float(cfg.get('guard_threshold',.985)):
        metrics.update(kept_aligned_input=True,fusion_seconds=time.perf_counter()-started)
        return feather_blend(registration),metrics
    if cfg.get('flow',False):b,flow=dense_align(registration);metrics.update(flow)
    mode=cfg.get('mode','graphcut')
    if mode=='adaptive_detail':
        qa,energy_a=source_clarity(a,registration.overlap_mask,with_energy=True)
        qb,energy_b=source_clarity(b,registration.overlap_mask,with_energy=True)
        ratio=(qa+.01)/(qb+.01)
        metrics.update(clarity_a=qa,clarity_b=qb,clarity_ratio=ratio,
                       clarity_energy_a=energy_a, clarity_energy_b=energy_b)
        margin=float(cfg.get('clarity_margin',1.1))
        preferred=None if 1/margin<=ratio<=margin else ratio>1
        if min(energy_a, energy_b) < float(cfg.get('min_clarity_energy', 0.0)):
            preferred = None
            metrics['low_texture_guard_applied'] = True
        if cfg.get('noise_guard',True):
            na=noise_floor(a,registration.overlap_mask);nb=noise_floor(b,registration.overlap_mask)
            metrics.update(noise_a=na,noise_b=nb)
            high,low=max(na,nb),min(na,nb)
            # Squared L2 norm of G(sigma=.6)-G(sigma=1.6): white-noise energy
            # transferred into the same frequency band used to measure clarity.
            noisy_energy=energy_a if na>nb else energy_b
            fraction=high**2*.167054/max(noisy_energy,.01)
            metrics['estimated_noise_fraction']=float(fraction)
            if (high>6 or fraction>.5) and high>2.5*max(low,.25):
                noisy_a=na>nb
                explained=blur_explanation(a if noisy_a else b,b if noisy_a else a,registration.overlap_mask)
                metrics['blur_explanation_rmse']=explained
                quiet=cv2.cvtColor(b if noisy_a else a,cv2.COLOR_BGR2GRAY)
                signal=float(np.std(quiet[registration.overlap_mask>0]))
                relative=explained/max(signal,.25)
                metrics['blur_explanation_relative_error']=relative
                if explained>max(.45,1.5*low) or relative>.15:
                    preferred=not noisy_a;metrics['noise_guard_applied']=True
        smooth_fraction, smooth_magnitude = low_frequency_photometric_disagreement(a, b, registration.overlap_mask)
        metrics.update(smooth_photometric_fraction=smooth_fraction,
                       smooth_photometric_magnitude=smooth_magnitude)
        if (preferred is not None and smooth_fraction >= float(cfg.get('photometric_fraction_threshold', .9))
                and smooth_magnitude >= float(cfg.get('photometric_magnitude_threshold', 2.5))
                and agreement < float(cfg.get('photometric_agreement_threshold', .95))):
            preferred = None
            metrics['photometric_guard_applied'] = True
        if preferred is None:
            result=feather_blend(registration);metrics['selection']='balanced; preserve feather'
        else:
            prefer_a=preferred
            owner=registration.mask_a if prefer_a else registration.mask_b
            distance=cv2.distanceTransform((owner>0).astype(np.uint8),cv2.DIST_L2,5)
            alpha=np.clip(distance/float(cfg.get('transition_width',16)),0,1)
            if not prefer_a:alpha=1-alpha
            result=detail_compose(registration,a,b,None,float(cfg.get('detail_scale',9)),alpha=alpha)
            metrics['selection']='a' if prefer_a else 'b'
    elif mode=='feather':result=feather_blend(replace(registration,canvas_b=b))
    else:
        if masks is None:masks=graphcut_masks(registration,a,b)
        bands=int(cfg.get('bands',3))
        if mode=='detail':
            result=detail_compose(registration,a,b,masks,float(cfg.get('detail_scale',5)))
        elif bands:
            masks=[cv2.bitwise_and(cv2.dilate(m,np.ones((7,7),np.uint8)),v)
                   for m,v in zip(masks,(registration.mask_a,registration.mask_b))]
            result=multiband(a,b,masks,bands)
        else:
            weight=(masks[0]>0).astype(np.float32)
            both=(masks[0]>0)&(masks[1]>0);weight[both]=.5
            result=np.clip(np.rint(a.astype(float)*weight[:,:,None]+b.astype(float)*(1-weight[:,:,None])),0,255).astype(np.uint8)
    only_a=(registration.mask_a>0)&(registration.mask_b==0)
    only_b=(registration.mask_b>0)&(registration.mask_a==0)
    result[only_a]=a[only_a];result[only_b]=registration.canvas_b[only_b]
    result[(registration.mask_a|registration.mask_b)==0]=0
    metrics['fusion_seconds']=time.perf_counter()-started
    return result,metrics
