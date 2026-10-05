"""Compare frozen 0.4 and proposed detail-preserving learned repair."""
from pathlib import Path
import argparse
import hashlib
import json
import cv2
import numpy as np
import torch
from scripts.validate_clear_fusion import make_cases
from xiyuan_mvp.clear_fusion import fuse
from xiyuan_mvp.structural_repair import CLEAR, PREDICTION_REVISION, predict_fields, compose_fields
from xiyuan_mvp.neural_alignment import repair_alignment
from xiyuan_mvp.seam_mask import mask_from_binary
from xiyuan_mvp.benchmark import LPIPSEvaluator, masked_metrics
from xiyuan_mvp.image_io import write_image
from xiyuan_mvp.run_io import write_json

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='outputs/structural-development');p.add_argument('--cached',action='store_true');args=p.parse_args()
    out=ROOT/args.output;out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);cv2.setNumThreads(4)
    evaluator=LPIPSEvaluator();rows=[]
    cfg={'device':'cpu','max_size':512,'iterations':12,'consistency':1.5,'max_flow':24,'exposure':True}
    cases=make_cases(['astronaut','camera'],['local_warp','light_gradient','blur_noise','noise'],640)
    write_json(out/'protocol.json',{'scope':'development on previously used sources; known geometry, full overlap edit mask',
        'source_sha256':hashlib.sha256((ROOT/'xiyuan_mvp/structural_repair.py').read_bytes()).hexdigest(),'config':cfg})
    for info,reg,ref in cases:
        cid=info['id'];folder=out/cid;folder.mkdir(exist_ok=True)
        base,clear=fuse(reg,CLEAR)
        mask=mask_from_binary(reg.overlap_mask,reg.overlap_mask,12)
        cache=folder/'fields.npz'
        cache_valid=args.cached and cache.exists() and json.loads((folder/'flow.json').read_text()).get('prediction_revision')==PREDICTION_REVISION
        if cache_valid:
            with np.load(cache) as z:fields=[z['forward'],z['reverse']]
            flow_metrics=json.loads((folder/'flow.json').read_text())
        else:
            fields,flow_metrics=predict_fields(reg,cfg);np.savez_compressed(cache,forward=fields[0],reverse=fields[1]);write_json(folder/'flow.json',flow_metrics)
        images={'clear04':base};metrics={'clear04':clear}
        if args.cached and (folder/'raft04.png').exists():
            from xiyuan_mvp.image_io import read_image
            images['raft04']=read_image(folder/'raft04.png')
        else:images['raft04'],metrics['raft04']=repair_alignment(reg,base,mask,cfg,method='raft_large')
        for strategy in ['midpoint','anchor']:
            for guard in [True,False]:
                name=strategy+('_guard' if guard else '')
                images[name],metrics[name]=compose_fields(reg,base,mask,fields,{**cfg,'only_balanced':guard},strategy=strategy)
        valid=np.full(ref.shape[:2],255,np.uint8);valid[:3]=0;valid[-3:]=0;valid[:,:3]=0;valid[:,-3:]=0
        scores={}
        for name,im in images.items():
            write_image(folder/(name+'.png'),im)
            scores[name]={**masked_metrics(im,ref,valid),'lpips':evaluator.evaluate(im,ref,valid)}
        for name,im in [('a',reg.image_a),('b',reg.image_b),('reference',ref)]:write_image(folder/(name+'.png'),im)
        rows.append({**info,'quality':scores,'metrics':metrics})
        write_json(out/'results.json',{'records':rows})
        print(cid,{k:round(v['lpips'],5) for k,v in scores.items()},flush=True)


if __name__=='__main__':main()
