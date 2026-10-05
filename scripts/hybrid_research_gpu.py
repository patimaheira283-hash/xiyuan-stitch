"""Regression experiment for the shipped neural + diffusion pipeline.

References are used only by quality(), never by the repair algorithm.
The four texture sources have already been used for development in v4.
"""
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
from scripts.research_gpu import inputs, ROOT, DATA, OUTPUT
from scripts.prepare_neural_validation import create_cases
from scripts.refined_research_gpu import quality
from xiyuan_mvp.benchmark import LPIPSEvaluator, make_comparison
from xiyuan_mvp.config import load_config
from xiyuan_mvp.image_io import write_image
from xiyuan_mvp.neural_alignment import repair_alignment
from xiyuan_mvp.pipeline import StitchPipeline
from xiyuan_mvp.run_io import environment, save_run, save_failure, write_json


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    records = []
    report = {'status': 'running', 'experiment': 'v5 hybrid pipeline regression',
              'protocol': {'split': 'development/regression, previously seen sources',
                           'references_used_by_repair': False,
                           'opacity': .1, 'local_illumination_default': False}}
    try:
        assert torch.cuda.is_available()
        torch.set_num_threads(4); cv2.setNumThreads(4)
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1, 8*1024**3/total))
        report.update(gpu=torch.cuda.get_device_name(0), actual_vram_bytes=total,
                      allocator_limit_bytes=min(total, 8*1024**3), environment=environment())
        sources = json.loads((ROOT/'data/neural-validation/sources.json').read_text())['sources']
        target = DATA/'regression-textures'
        for source in sources:
            response = requests.get(source['url'], timeout=90); response.raise_for_status()
            assert hashlib.sha256(response.content).hexdigest() == source['sha256']
            p = target/source['file']; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(response.content)
        cases = create_cases(target, sources) + [c for c in inputs() if not c.get('reference')]
        for case in cases: case['split'] = 'development_regression'
        write_json(OUTPUT/'inputs-v5.json', {'cases': cases})
        cfg = load_config()
        cfg['registration'].update(method='loftr', loftr_device='cuda', loftr_max_size=640)
        cfg['refinement']['enabled'] = False
        cfg['neural_alignment'].update(device='cuda', method='raft_large')
        cfg['inpainting'].update(device='cuda', local_files_only=False, variant='fp16',
            cache_dir='/kaggle/temp/xiyuan-hf-cache', controlnet_scale=.8,
            steps=20, strength=.25, guidance_scale=3.5, controlnet='canny')
        cfg['blend']['ai_opacity'] = .1
        write_json(OUTPUT/'config-v5.json', cfg)
        evaluator = LPIPSEvaluator(); prepared = {}
        def record(row):
            records.append(row)
            write_json(OUTPUT/'progress-v5.json', {'records': records})
            print(row['profile'], row['id'], row['status'], row.get('error',''), flush=True)
        for case in cases:
            row = {'id': case['id'], 'profile': 'baseline', 'status': 'failed'}
            try:
                result = StitchPipeline(cfg).run(case['image_a'], case['image_b'])
                prepared[case['id']] = result
                save_run(OUTPUT/'baseline'/case['id'], result, cfg, extra={'case': case})
                row.update(status='success', metrics=result.metrics,
                    quality=quality({'Feather': result.traditional_image, 'Poisson': result.poisson_image}, case, result, evaluator))
            except Exception as exc: row['error'] = f'{type(exc).__name__}: {exc}'
            record(row)
        profiles = ['dis', 'raft_large', 'raft_local', 'hybrid_canny', 'hybrid_tile_lcm']
        pipe = StitchPipeline(deepcopy(cfg))
        for profile in profiles:
            for case in cases:
                folder = OUTPUT/profile/case['id']
                row = {'id': case['id'], 'profile': profile, 'status': 'failed'}
                try:
                    base = prepared[case['id']]
                    if profile.startswith('hybrid'):
                        pipe.config['inpainting'].update(engine='hybrid',
                            controlnet='tile' if profile.endswith('lcm') else 'canny',
                            use_lcm=profile.endswith('lcm'), steps=8 if profile.endswith('lcm') else 20,
                            strength=.5 if profile.endswith('lcm') else .25,
                            guidance_scale=1.0 if profile.endswith('lcm') else 3.5)
                        result = pipe.run(case['image_a'], case['image_b'], prepared=base, use_ai=True)
                        image, metrics = result.final_image, result.metrics
                        save_run(folder, result, pipe.config, extra={'case': case, 'profile': profile})
                    else:
                        image, metrics = repair_alignment(base.registration, base.traditional_image, base.mask,
                            {**cfg['neural_alignment'], 'local_illumination': profile=='raft_local'},
                            method='dis' if profile=='dis' else 'raft_large')
                        write_image(folder/'result.png', image)
                    outside = base.mask.soft_mask == 0
                    assert np.array_equal(image[outside], base.traditional_image[outside])
                    row.update(status='success', metrics=metrics, outside_mask_unchanged=True,
                               quality=quality({profile:image}, case, base, evaluator))
                    make_comparison(folder/'comparison.png', {'Feather':base.traditional_image,
                        'Poisson':base.poisson_image, profile:image}, base.mask.bbox)
                except Exception as exc:
                    row['error'] = f'{type(exc).__name__}: {exc}'
                    save_failure(folder, exc, pipe.config); traceback.print_exc()
                write_json(folder/'evaluation.json', row); record(row)
        report['status'] = 'completed' if all(r['status']=='success' for r in records) else 'completed_with_failures'
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report.update(records=records, elapsed_seconds=time.perf_counter()-started)
        write_json(OUTPUT/'report-v5.json', report)
        with zipfile.ZipFile('/kaggle/working/research-v5-bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
            for p in OUTPUT.rglob('*'):
                if p.is_file(): z.write(p, p.relative_to(OUTPUT.parent))


if __name__ == '__main__': main()
