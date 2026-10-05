"""Small development comparison of RAFT against identical non-neural preparations."""
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from skimage import data
from xiyuan_mvp.benchmark import LPIPSEvaluator,masked_metrics,make_comparison
from xiyuan_mvp.blending import poisson_blend
from xiyuan_mvp.image_io import read_image,write_image
from xiyuan_mvp.neural_alignment import repair_alignment
from xiyuan_mvp.run_io import load_run,write_json

ROOT=Path(__file__).resolve().parents[1]


def main(large=False,refined=False):
    torch.set_num_threads(4);cv2.setNumThreads(4)
    output=ROOT/('outputs/neural-repair-hybrid-development' if refined else 'outputs/neural-repair-large-development' if large else 'outputs/neural-repair-development');output.mkdir(exist_ok=True)
    dataset=ROOT/'outputs/research/v2/extracted/research-results'
    evaluator=LPIPSEvaluator();rows=[]
    ids=['brick_deformation_2','brick_deformation_3','grass_deformation_2','gravel_deformation_3','coffee_wood_deformation_2','coffee_wood_deformation_3']
    for cid in ids:
        r,d=load_run(dataset/'structural_only'/cid)
        reference=cv2.cvtColor(data.coffee(),cv2.COLOR_RGB2BGR) if cid.startswith('coffee') else read_image(ROOT/'data/benchmark/sources'/(cid.split('_')[0]+'.png'))
        h,w=r.final_image.shape[:2]
        ref=cv2.warpPerspective(reference,r.registration.canvas_transform,(w,h))
        valid=cv2.warpPerspective(np.full(reference.shape[:2],255,np.uint8),r.registration.canvas_transform,(w,h));valid=cv2.erode(valid,np.ones((7,7),np.uint8));mask=cv2.bitwise_and(valid,r.mask.binary_mask)
        images={'Feather':r.traditional_image,'Poisson':poisson_blend(r.registration)}
        for method in ['exposure_only','dis','raft_refined' if refined else 'raft_large' if large else 'raft']:
            result,metrics=repair_alignment(r.registration,r.traditional_image,r.mask,{'device':'cpu','max_size':512,'iterations':12},method=method)
            images[method]=result
            assert np.array_equal(result[r.mask.soft_mask==0],r.traditional_image[r.mask.soft_mask==0])
            write_json(output/cid/(method+'-runtime.json'),metrics)
        scores={}
        for name,image in images.items():
            scores[name]={**masked_metrics(image,ref,mask),'lpips':evaluator.evaluate(image,ref,mask)}
            write_image(output/cid/(name+'.png'),image)
        make_comparison(output/cid/'comparison.png',images,r.mask.bbox)
        rows.append({'id':cid,'quality':scores});write_json(output/'results.json',{'records':rows})
        print(cid,{k:round(v['lpips'],5) for k,v in scores.items()},flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--large',action='store_true');p.add_argument('--refined',action='store_true');args=p.parse_args();main(args.large,args.refined)
