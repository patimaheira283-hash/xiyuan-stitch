"""Recompute the corrected baseline; keep original artifacts and all AI outputs intact."""
from dataclasses import replace
import json
from pathlib import Path
import statistics
import cv2
import numpy as np
import torch

from xiyuan_mvp.benchmark import LPIPSEvaluator, masked_metrics, make_comparison
from xiyuan_mvp.blending import poisson_blend
from xiyuan_mvp.image_io import read_image, write_image
from xiyuan_mvp.run_io import load_run, write_json

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'outputs/research/v2/extracted/research-results'
OUT=ROOT/'outputs/research/poisson-correction-v2'


def main():
    torch.set_num_threads(4);cv2.setNumThreads(4)
    evaluator=LPIPSEvaluator()
    cases=json.loads((DATA/'inputs-v2.json').read_text(encoding='utf-8'))['cases']
    old=json.loads((DATA/'report-v2.json').read_text(encoding='utf-8'))
    old_rows={r['id']:r for r in old['records'] if r.get('profile')=='structural_only'}
    rows=[]
    for case in cases:
        baseline,report=load_run(DATA/'structural_only'/case['id'])
        corrected=poisson_blend(baseline.registration)
        folder=OUT/case['id'];write_image(folder/'poisson_corrected.png',corrected)
        row={'id':case['id'],'quality':None,'previous_baseline_invalid':True}
        if case.get('reference'):
            source=case['source_id']
            if source=='coffee_wood':
                from skimage import data
                ref=cv2.cvtColor(data.coffee(),cv2.COLOR_RGB2BGR)
            else:ref=read_image(ROOT/'data/benchmark/sources'/f'{source}.png')
            a=baseline.registration.image_a
            # The saved A crop is byte-identical to the reference's left crop.
            # This verifies the regenerated public reference matches the cloud input.
            assert np.array_equal(a,ref[:a.shape[0],:a.shape[1]]),f'Reference differs: {case["id"]}'
            h,w=corrected.shape[:2]
            transform=baseline.registration.canvas_transform@np.asarray(case['reference_to_a'])
            reference=cv2.warpPerspective(ref,transform,(w,h))
            valid=cv2.warpPerspective(np.full(ref.shape[:2],255,np.uint8),transform,(w,h))
            valid=cv2.erode(valid,np.ones((7,7),np.uint8));seam=cv2.bitwise_and(valid,baseline.mask.binary_mask)
            row['quality']={region:{**masked_metrics(corrected,reference,mask),'lpips':evaluator.evaluate(corrected,reference,mask)}
                            for region,mask in [('full',valid),('seam',seam)]}
        ai=read_image(DATA/'refined_canny'/case['id']/'07_ai_result.png')
        make_comparison(folder/'comparison.png',{'Feather':baseline.traditional_image,'Correct Poisson':corrected,
                        'Structural':baseline.repair_base_image,'AI':ai},baseline.mask.bbox)
        rows.append(row);write_json(OUT/'progress.json',{'records':rows});print(case['id'],row['quality'],flush=True)
    valid=[r for r in rows if r['quality']]
    summary={'records':rows,'fix':'pass a copy of the clone mask to OpenCV because it modifies the argument in place',
             'corrected_poisson_mean_lpips':{region:statistics.mean(r['quality'][region]['lpips'] for r in valid) for region in ('full','seam')}}
    score={r['id']:r['quality'] for r in valid}
    summary['ai_wins_over_correct_poisson']={p:{region:sum(r['quality'][p][region]['lpips']<score[r['id']][region]['lpips'] for r in old['records']
        if r.get('profile')==p and r.get('quality')) for region in ('full','seam')} for p in ('raw_canny','refined_canny','refined_tile','refined_tile_lcm')}
    write_json(OUT/'audit.json',summary);print(json.dumps({k:v for k,v in summary.items() if k!='records'},indent=2),flush=True)


if __name__=='__main__':main()
