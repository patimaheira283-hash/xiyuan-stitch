"""Balanced unseen-source composition checks and pinned real-photo outputs.

The clean panorama is never provided to the algorithm. Known crop transforms
isolate fusion from feature matching; those rows are not registration benchmarks.
"""
from pathlib import Path
import hashlib
import json
import time
import argparse
import cv2
import numpy as np
import torch
from skimage import data
from xiyuan_mvp.benchmark import LPIPSEvaluator,masked_metrics
from xiyuan_mvp.blending import feather_blend,poisson_blend
from xiyuan_mvp.clear_fusion import fuse,graphcut_masks
from xiyuan_mvp.image_io import write_image
from xiyuan_mvp.run_io import write_json,load_run
from xiyuan_mvp.types import RegistrationResult

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'outputs/clear-fusion-validation'
PROFILES={'graphcut_hard':{'bands':0},'clear_detail':{'mode':'detail','detail_scale':9,'guard':True}}


def make_cases(names=None,variants=None,max_size=None):
    cases=[]
    for name in names or ['astronaut','chelsea','camera','immunohistochemistry']:
        rgb=getattr(data,name)()
        if rgb.ndim==2:rgb=cv2.cvtColor(rgb,cv2.COLOR_GRAY2RGB)
        ref=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
        if max_size and max(ref.shape[:2])>max_size:
            ref=cv2.resize(ref,None,fx=max_size/max(ref.shape[:2]),fy=max_size/max(ref.shape[:2]),interpolation=cv2.INTER_AREA)
        h,w=ref.shape[:2];cut=int(w*.70);start=int(w*.30)
        for variant in variants or ['local_warp','light_gradient','blur_noise']:
            for damaged_side in ['a','b']:
                a=ref[:,:cut].copy();b=ref[:,start:].copy()
                damaged=a if damaged_side=='a' else b
                ih,iw=damaged.shape[:2];yy,xx=np.indices((ih,iw),dtype=np.float32)
                if variant=='local_warp':
                    center=(start+cut)/2-(0 if damaged_side=='a' else start)
                    displacement=8*np.exp(-((xx-center)/(w*.12))**2)*np.sin(yy/(h/8)+.37)
                    changed=cv2.remap(damaged,xx,yy+displacement,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
                elif variant=='light_gradient':changed=np.clip(damaged.astype(float)*np.linspace(.75,1.25,h)[:,None,None],0,255).astype(np.uint8)
                elif variant=='noise':
                    changed=np.clip(damaged.astype(float)+np.random.default_rng(8546).normal(0,14,damaged.shape),0,255).astype(np.uint8)
                else:
                    changed=np.clip(cv2.GaussianBlur(damaged,(0,0),1.2).astype(float)+np.random.default_rng(7643).normal(0,2,damaged.shape),0,255).astype(np.uint8)
                if damaged_side=='a':a=changed
                else:b=changed
                ca=np.zeros_like(ref);cb=ca.copy();ca[:,:cut]=a;cb[:,start:]=b
                ma=np.zeros((h,w),np.uint8);mb=ma.copy();ma[:,:cut]=255;mb[:,start:]=255
                overlap=cv2.bitwise_and(ma,mb)
                reg=RegistrationResult(a,b,ca,cb,ma,mb,overlap,
                    np.array([[1,0,start],[0,1,0],[0,0,1]],float),np.eye(3),np.zeros((8,8,3),np.uint8),
                    {'feature_method':'known_crop_transform; isolates composition'})
                cases.append(({'id':name+'_'+variant+'_'+damaged_side,'source':name,'variant':variant,
                    'damaged_side':damaged_side,'reference_sha256':hashlib.sha256(ref.tobytes()).hexdigest()},reg,ref))
    return cases


def main():
    global OUTPUT,PROFILES
    p=argparse.ArgumentParser();p.add_argument('--adaptive',action='store_true');p.add_argument('--holdout',action='store_true');p.add_argument('--fresh',action='store_true');p.add_argument('--regression',action='store_true');p.add_argument('--output');args=p.parse_args()
    if args.fresh:args.holdout=True
    if args.holdout:args.adaptive=True
    if args.adaptive:
        OUTPUT=ROOT/'outputs/clear-adaptive-balanced';PROFILES={'adaptive':{'mode':'adaptive_detail','detail_scale':9,'guard':True}}
    if args.holdout:OUTPUT=ROOT/'outputs/clear-adaptive-heldout'
    if args.output:OUTPUT=ROOT/args.output
    OUTPUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);cv2.setNumThreads(4)
    names=['text','coins','hubble_deep_field','moon'] if args.holdout else ['astronaut','chelsea','camera','immunohistochemistry']
    if args.fresh:names=['page','retina','cell','clock']
    variants=['local_warp','light_gradient','blur_noise','noise'] if args.holdout else None
    cases=make_cases(names,variants,640 if args.holdout else None)
    write_json(OUTPUT/'protocol.json',{'profiles':PROFILES,'cases':[x[0] for x in cases],
        'scope':('regression on previously revealed cases' if args.regression else
                 'unseen sources, noise-aware algorithm frozen before results' if args.fresh else
                 'regression on previously revealed cases' if args.output else
                 'unseen sources and a new high-noise stressor; frozen algorithm' if args.holdout else
                 'balanced regression after v1 source-choice failure' if args.adaptive else 'initial unseen natural sources')+'; known geometry; composition evaluation only',
        'algorithm_sha256':hashlib.sha256((ROOT/'xiyuan_mvp/clear_fusion.py').read_bytes()).hexdigest(),
        'source_documentation':{n:getattr(data,n).__doc__ for n in names}})
    evaluator=LPIPSEvaluator();rows=[]
    for info,reg,ref in cases:
        folder=OUTPUT/info['id'];images={'Feather':feather_blend(reg),'Poisson':poisson_blend(reg)};metrics={}
        masks=None if args.adaptive else graphcut_masks(reg,reg.canvas_a,reg.canvas_b)
        for name,cfg in PROFILES.items():images[name],metrics[name]=fuse(reg,cfg,masks=masks)
        valid=np.full(ref.shape[:2],255,np.uint8);valid[:3]=0;valid[-3:]=0;valid[:,:3]=0;valid[:,-3:]=0
        scores={}
        for name,image in images.items():
            write_image(folder/(name+'.png'),image)
            scores[name]={**masked_metrics(image,ref,valid),'lpips':evaluator.evaluate(image,ref,valid)}
        write_image(folder/'reference.png',ref);write_image(folder/'a.png',reg.image_a);write_image(folder/'b.png',reg.image_b)
        rows.append({**info,'quality':scores,'metrics':metrics});write_json(OUTPUT/'results.json',{'records':rows,'profiles':PROFILES})
        print(info['id'],{n:round(v['lpips'],5) for n,v in scores.items()},flush=True)
    # Preselected previously saved real scenes: no reference-based score is invented.
    real=['spw_10_efab8e6e','lpc_library_dd718dea','rew_rew_wall_de776f7d','ges_cave_01_atrium_bad372c8']
    for cid in real:
        r,_=load_run(ROOT/'outputs/research/v4/extracted/research-results/baseline'/cid)
        image,metrics=fuse(r.registration,next(iter(PROFILES.values())))
        write_image(OUTPUT/'real'/cid/'clear_detail.png',image)
        write_image(OUTPUT/'real'/cid/'Feather.png',r.traditional_image)
        write_json(OUTPUT/'real'/cid/'metrics.json',{'id':cid,'metrics':metrics,'quality':None,'reason':'No reference panorama'})
    print('Balanced validation and four real-scene outputs completed',flush=True)


if __name__=='__main__':main()
