"""Evaluate input-preserving neural repair against corrected traditional baselines."""
from copy import deepcopy
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

from scripts.research_gpu import inputs,ROOT,DATA,OUTPUT
from scripts.deformation_cases import create_cases as deformation_cases
from scripts.prepare_neural_validation import create_cases as new_cases
from scripts.refined_research_gpu import quality
from scripts.final_research_gpu import cases as full_cases
from xiyuan_mvp.benchmark import LPIPSEvaluator,make_comparison
from xiyuan_mvp.config import load_config
from xiyuan_mvp.image_io import read_image,write_image
from xiyuan_mvp.neural_alignment import repair_alignment
from xiyuan_mvp.pipeline import StitchPipeline
from xiyuan_mvp.run_io import environment,save_run,save_failure,write_json


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();records=[]
    report={'status':'running','experiment':'neural repair validation v4; corrected Poisson; four new CC0 texture sources',
        'protocol':{'max_size':512,'iterations':12,'consistency':1.5,'max_flow':24,'exposure':True,
                    'algorithms':['exposure_only','dis','raft_large','raft_refined'],
                    'new_sources':'four Poly Haven assets selected before seeing results, 28 derived cases'}}
    try:
        assert torch.cuda.is_available();torch.set_num_threads(4);cv2.setNumThreads(4)
        total=torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1,8*1024**3/total))
        report.update(gpu=torch.cuda.get_device_name(0),actual_vram_bytes=total,allocator_limit_bytes=min(total,8*1024**3),environment=environment())
        sources=json.loads((ROOT/'data/neural-validation/sources.json').read_text(encoding='utf-8'))['sources']
        new_root=DATA/'new-textures'
        for source in sources:
            response=requests.get(source['url'],timeout=90);response.raise_for_status()
            if hashlib.sha256(response.content).hexdigest()!=source['sha256']:raise ValueError('Source SHA256 mismatch')
            p=new_root/source['file'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(response.content)
        # Retain previous development cases. All 30 previously selected real
        # scenes are also evaluated, with development overlap explicitly marked.
        cases=[c for c in inputs() if c.get('reference')]+deformation_cases(DATA/'deformation')+new_cases(new_root,sources)
        cases += [c for c in full_cases() if not c.get('reference')]
        write_json(OUTPUT/'inputs-v4.json',{'cases':cases,'config':report['protocol']})
        cfg=load_config();cfg['registration'].update(method='loftr',loftr_device='cuda',loftr_max_size=640)
        cfg['refinement']['enabled']=False
        cfg['neural_alignment'].update(device='cuda',max_size=512,iterations=12,consistency=1.5,max_flow=24,exposure=True)
        evaluator=LPIPSEvaluator();prepared={}
        for case in cases:
            row={'id':case['id'],'profile':'baseline','split':case['split'],'status':'failed'}
            try:
                r=StitchPipeline(cfg).run(case['image_a'],case['image_b']);prepared[case['id']]=r
                save_run(OUTPUT/'baseline'/case['id'],r,cfg,extra={'case':case})
                row.update(status='success',metrics=r.metrics,quality=quality({'Feather':r.traditional_image,'Poisson':r.poisson_image},case,r,evaluator))
            except Exception as exc:row['error']=f'{type(exc).__name__}: {exc}'
            records.append(row);write_json(OUTPUT/'progress-v4.json',{'records':records});print('BASE',case['id'],row['status'],flush=True)
        for method in ['exposure_only','dis','raft_large','raft_refined']:
            for case in cases:
                folder=OUTPUT/method/case['id']
                row={'id':case['id'],'profile':method,'split':case['split'],'status':'failed'}
                try:
                    if case['id'] not in prepared:raise ValueError('Global alignment failed')
                    r=prepared[case['id']]
                    output,metrics=repair_alignment(r.registration,r.traditional_image,r.mask,cfg['neural_alignment'],method=method)
                    outside=r.mask.soft_mask==0
                    assert np.array_equal(output[outside],r.traditional_image[outside])
                    row.update(status='success',metrics=metrics,quality=quality({method:output},case,r,evaluator),outside_mask_unchanged=True)
                    write_image(folder/'result.png',output)
                    make_comparison(folder/'comparison.png',{'Feather':r.traditional_image,'Poisson':r.poisson_image,method:output},r.mask.bbox)
                except Exception as exc:row['error']=f'{type(exc).__name__}: {exc}';save_failure(folder,exc,cfg)
                records.append(row);write_json(folder/'evaluation.json',row);write_json(OUTPUT/'progress-v4.json',{'records':records})
                print(method,case['id'],row['status'],flush=True)
        report['status']='completed_with_failures' if any(r['status']!='success' for r in records) else 'completed'
    except Exception as exc:report.update(status='failed',error=f'{type(exc).__name__}: {exc}');traceback.print_exc();raise
    finally:
        report.update(records=records,elapsed_seconds=time.perf_counter()-started)
        write_json(OUTPUT/'report-v4.json',report)
        with zipfile.ZipFile('/kaggle/working/research-v4-bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
            for p in OUTPUT.rglob('*'):
                if p.is_file():z.write(p,p.relative_to(OUTPUT.parent))


if __name__=='__main__':main()
