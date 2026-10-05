"""Evaluate new whole-overlap fusion against frozen 0.3 baselines on seen data."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from xiyuan_mvp.clear_fusion import fuse,graphcut_masks
from xiyuan_mvp.benchmark import LPIPSEvaluator,masked_metrics
from xiyuan_mvp.image_io import read_image,write_image
from xiyuan_mvp.run_io import load_run,write_json

ROOT=Path(__file__).resolve().parents[1]
PROFILES={'graphcut_hard':{'bands':0},'graphcut_3':{'bands':3},'graphcut_5':{'bands':5},
          'flow_feather':{'flow':True,'mode':'feather'},'flow_graphcut':{'flow':True,'bands':3}}


def main():
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=28);p.add_argument('--output',default='outputs/clear-fusion-development');p.add_argument('--detail',action='store_true');p.add_argument('--adaptive',action='store_true')
    args=p.parse_args();output=ROOT/args.output;output.mkdir(parents=True,exist_ok=True)
    profiles=PROFILES if not args.detail else {'detail_3':{'mode':'detail','detail_scale':3,'guard':True},
        'detail_9':{'mode':'detail','detail_scale':9,'guard':True},'flow_detail':{'mode':'detail','detail_scale':9,'flow':True,'guard':True}}
    if args.adaptive:profiles={'adaptive':{'mode':'adaptive_detail','detail_scale':9,'guard':True}}
    cv2.setNumThreads(4);torch.set_num_threads(4);evaluator=LPIPSEvaluator();rows=[]
    cases=json.loads((ROOT/'data/neural-validation/manifest.json').read_text())['cases'][:args.limit]
    for case in cases:
        result,_=load_run(ROOT/'outputs/research/v4/extracted/research-results/baseline'/case['id'])
        ref0=read_image(case['reference']);h,w=result.final_image.shape[:2]
        ref=cv2.warpPerspective(ref0,result.registration.canvas_transform,(w,h))
        valid=cv2.warpPerspective(np.full(ref0.shape[:2],255,np.uint8),result.registration.canvas_transform,(w,h))
        valid=cv2.erode(valid,np.ones((7,7),np.uint8));seam=cv2.bitwise_and(valid,result.mask.binary_mask)
        images={'Feather':result.traditional_image,'Poisson':result.poisson_image}
        metrics={}
        shared_masks=None
        for name,config in profiles.items():
            if not args.adaptive and not config.get('flow') and shared_masks is None:shared_masks=graphcut_masks(result.registration,result.registration.canvas_a,result.registration.canvas_b)
            images[name],metrics[name]=fuse(result.registration,config,masks=shared_masks if not config.get('flow') else None)
        scores={}
        for name,image in images.items():
            write_image(output/case['id']/(name+'.png'),image)
            scores[name]={reg:{**masked_metrics(image,ref,mask),'lpips':evaluator.evaluate(image,ref,mask)}
                          for reg,mask in [('full',valid),('seam',seam)]}
        rows.append({'id':case['id'],'variant':case['variant'],'scores':scores,'metrics':metrics})
        write_json(output/'results.json',{'split':'seen development sources; whole-overlap fusion',
            'profiles':profiles,'records':rows})
        print(case['id'],{n:round(v['full']['lpips'],5) for n,v in scores.items()},flush=True)


if __name__=='__main__':main()
