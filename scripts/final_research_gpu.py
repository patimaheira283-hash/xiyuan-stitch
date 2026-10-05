"""Frozen evaluation: 30 real pairs plus 24 unseen perturbation variants of known sources."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
import traceback
import zipfile

import cv2
import numpy as np
import requests
import torch

from scripts.research_gpu import ROOT, DATA, OUTPUT
from scripts.refined_research_gpu import quality
from scripts.high_resolution_acceptance import create_scene
from xiyuan_mvp.benchmark import LPIPSEvaluator, make_comparison
from xiyuan_mvp.config import load_config
from xiyuan_mvp.dataset import create_dataset
from xiyuan_mvp.image_io import read_image, write_image
from xiyuan_mvp.pipeline import StitchPipeline
from xiyuan_mvp.run_io import environment, save_run, save_failure, write_json

PROFILES={
    'raw_canny':{'controlnet':'canny','use_lcm':False,'steps':20,'strength':0.25,'guidance_scale':3.5},
    'refined_canny':{'controlnet':'canny','use_lcm':False,'steps':20,'strength':0.25,'guidance_scale':3.5},
    'refined_tile_lcm':{'controlnet':'tile','use_lcm':True,'steps':8,'strength':0.5,'guidance_scale':1.0},
}


def cases():
    folder=DATA/'final-controlled'
    path=create_dataset(folder)
    result=[]
    for c in json.loads(path.read_text())['cases']:
        if c['variant'] in ('brighter','mixed'):continue
        result.append({**c,'image_a':str(folder/c['image_a']),'image_b':str(folder/c['image_b']),
            'reference':str(folder/c['reference']),'split':'held-out perturbations of development sources; not independent scenes'})
    counts={}
    for c in json.loads((ROOT/'data/research-cases.json').read_text(encoding='utf-8'))['cases']:
        counts[c['category']]=counts.get(c['category'],0)+1
        local=[]
        for index,item in enumerate(c['files']):
            r=requests.get(item['url'],timeout=90);r.raise_for_status()
            if hashlib.sha256(r.content).hexdigest()!=item['sha256']:raise ValueError('Input hash mismatch')
            p=DATA/'final-real'/c['id']/f'{index}.img';p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(r.content)
            im=read_image(p);h,w=im.shape[:2]
            if max(h,w)>1600:im=cv2.resize(im,None,fx=1600/max(h,w),fy=1600/max(h,w),interpolation=cv2.INTER_AREA)
            p=p.with_suffix('.png');write_image(p,im);local.append(str(p))
        result.append({**c,'image_a':local[0],'image_b':local[1],
            'split':'development scene repeated at 1600 cap' if counts[c['category']]<=2 else 'held-out scene',
            'resizing':'long side capped at 1600; author bytes SHA256 checked'})
    return result


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();records=[]
    report={'status':'running','experiment':'frozen v3: 30 real scenes, 24 held-out perturbations, fixed v2 profiles; no hyperparameter search'}
    try:
        assert torch.cuda.is_available()
        total=torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1,8*1024**3/total))
        torch.set_num_threads(4)
        cv2.setNumThreads(4)
        report.update(gpu=torch.cuda.get_device_name(0),actual_vram_bytes=total,allocator_limit_bytes=min(total,8*1024**3),
            environment=environment(),budget_note='8 GiB allocator cap on T4; not native 8GB hardware')
        cfg=load_config();cfg['registration'].update(method='loftr',loftr_device='cuda',loftr_max_size=640)
        cfg['refinement']['enabled']=True;cfg['blend']['ai_opacity']=0.2
        cfg['inpainting'].update(device='cuda',local_files_only=False,cache_dir='/kaggle/temp/xiyuan-hf-cache',controlnet_scale=0.8)
        selected=cases();write_json(OUTPUT/'inputs-v3.json',{'cases':selected,'profiles':PROFILES,'config':cfg})
        evaluator=LPIPSEvaluator();prepared={}
        for case in selected:
            row={'id':case['id'],'profile':'structural_only','split':case['split'],'status':'failed','registration_comparison':{}}
            try:
                for method in ('sift','loftr'):
                    local=deepcopy(cfg);local['registration']['method']=method
                    try:
                        baseline=StitchPipeline(local).run(case['image_a'],case['image_b'])
                        row['registration_comparison'][method]={'status':'success',**baseline.metrics}
                        if method=='loftr':prepared[case['id']]=baseline
                    except Exception as e:row['registration_comparison'][method]={'status':'failed','error':str(e)}
                baseline=prepared[case['id']]
                row.update(status='success',metrics=baseline.metrics,quality=quality({'Feather':baseline.traditional_image,
                    'Poisson':baseline.poisson_image,'Structural':baseline.repair_base_image},case,baseline,evaluator))
                save_run(OUTPUT/'structural_only'/case['id'],baseline,cfg,extra={'case':case})
            except Exception as exc:row['error']=f'{type(exc).__name__}: {exc}'
            records.append(row);write_json(OUTPUT/'progress-v3.json',{'records':records});print('BASE',case['id'],row['status'],flush=True)
        pipe=StitchPipeline(cfg)
        for profile,options in PROFILES.items():
            pipe.config['inpainting'].update(options)
            for case in selected:
                folder=OUTPUT/profile/case['id']
                row={'id':case['id'],'profile':profile,'split':case['split'],'status':'failed'}
                try:
                    if case['id'] not in prepared:raise ValueError('LoFTR registration failed; AI not run')
                    baseline=prepared[case['id']]
                    if profile=='raw_canny':baseline=replace(baseline,repair_base_image=baseline.traditional_image,metrics={**baseline.metrics,'refinement_enabled':False})
                    result=pipe.run(case['image_a'],case['image_b'],prepared=baseline,use_ai=True)
                    assert not result.metrics.get('test_double')
                    outside=result.mask.soft_mask==0
                    assert np.array_equal(result.final_image[outside],baseline.repair_base_image[outside])
                    row.update(status='success',metrics=result.metrics,quality=quality({profile:result.final_image},case,result,evaluator),
                        outside_mask_unchanged=True,ai_opacity=0.2)
                    # Baseline keeps full reproducible geometry. Each profile keeps its actual
                    # result, config, and evaluation without repeating all input PNGs three times.
                    write_image(folder/'result.png',result.final_image)
                    write_json(folder/'config.json',pipe.config)
                    make_comparison(folder/'comparison.png',{'Feather':baseline.traditional_image,'Poisson':baseline.poisson_image,
                        'Structural':prepared[case['id']].repair_base_image,profile:result.final_image},result.mask.bbox)
                except Exception as exc:
                    row['error']=f'{type(exc).__name__}: {exc}';save_failure(folder,exc,pipe.config)
                write_json(folder/'evaluation.json',row);records.append(row)
                write_json(OUTPUT/'progress-v3.json',{'records':records});print(profile,case['id'],row['status'],flush=True)
        # Engineering stress: three inputs, original 4K-wide canvas, full LoFTR+ControlNet+LCM mode.
        try:
            scene=create_scene();sequence=[scene[:,:2200],scene[:,1100:3300],scene[:,2200:]]
            result=pipe.run_many(sequence,use_ai=True)
            assert result.final_image.shape[1]>=4090
            save_run(OUTPUT/'high-resolution',result,pipe.config)
            report['high_resolution']={'status':'success','metrics':result.metrics,'shape':list(result.final_image.shape),'kind':'procedural engineering stress, not quality evidence'}
        except Exception as e:report['high_resolution']={'status':'failed','error':f'{type(e).__name__}: {e}'}
        report['status']='completed_with_failures' if any(r['status']!='success' for r in records) else 'completed'
    except Exception as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}');traceback.print_exc();raise
    finally:
        report.update(records=records,elapsed_seconds=time.perf_counter()-started)
        write_json(OUTPUT/'report-v3.json',report)
        with zipfile.ZipFile('/kaggle/working/research-v3-bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
            for p in OUTPUT.rglob('*'):
                if p.is_file():z.write(p,p.relative_to(OUTPUT.parent))


if __name__=='__main__':main()
