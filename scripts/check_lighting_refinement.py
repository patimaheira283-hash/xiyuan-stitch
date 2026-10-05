import json
from pathlib import Path
import cv2
import numpy as np
import torch
from xiyuan_mvp.neural_alignment import repair_alignment
from xiyuan_mvp.benchmark import LPIPSEvaluator,masked_metrics,make_comparison
from xiyuan_mvp.image_io import read_image,write_image
from xiyuan_mvp.run_io import load_run,write_json

ROOT=Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(4);cv2.setNumThreads(4);evaluator=LPIPSEvaluator()
    output=ROOT/'outputs/lighting-refinement';output.mkdir(exist_ok=True)
    cases=json.loads((ROOT/'data/neural-validation/manifest.json').read_text(encoding='utf-8'))['cases'];rows=[]
    for case in cases:
        r,_=load_run(ROOT/'outputs/research/v4/extracted/research-results/baseline'/case['id'])
        ref0=read_image(case['reference']);h,w=r.final_image.shape[:2]
        ref=cv2.warpPerspective(ref0,r.registration.canvas_transform,(w,h))
        valid=cv2.warpPerspective(np.full(ref0.shape[:2],255,np.uint8),r.registration.canvas_transform,(w,h));valid=cv2.erode(valid,np.ones((7,7),np.uint8));mask=cv2.bitwise_and(valid,r.mask.binary_mask)
        scores={};images={'Feather':r.traditional_image}
        for name,method,local in [('global_only','exposure_only',False),('local_only','exposure_only',True),('raft_local','raft_large',True)]:
            image,metrics=repair_alignment(r.registration,r.traditional_image,r.mask,{'device':'cpu','local_illumination':local},method=method)
            images[name]=image;write_image(output/case['id']/(name+'.png'),image)
        for name,image in images.items():scores[name]={**masked_metrics(image,ref,mask),'lpips':evaluator.evaluate(image,ref,mask)}
        make_comparison(output/case['id']/'comparison.png',images,r.mask.bbox)
        rows.append({'id':case['id'],'variant':case['variant'],'quality':scores})
        write_json(output/'results.json',{'records':rows})
        print(case['id'],{n:round(v['lpips'],5) for n,v in scores.items()},flush=True)


if __name__=='__main__':main()
