"""Paired clear/RAFT/diffusion ablation with exact shared geometry and generated pixels.

Old feather arithmetic is replayed only for diagnosis. Reference images never
enter inference, margin selection, or the runtime gate. All sources are development
sources; dual damage is a stress extension, not independent held-out photography.
"""
from copy import deepcopy
from dataclasses import replace
import time
import traceback
import zipfile

import cv2
import numpy as np
import torch
from skimage import data

from scripts.research_gpu import inputs, OUTPUT, DATA, ROOT
from scripts.deformation_cases import create_cases
from scripts.refined_research_gpu import quality
from xiyuan_mvp.ai_guard import apply_ai_guard
from xiyuan_mvp.benchmark import LPIPSEvaluator
from xiyuan_mvp.blending import feather_blend, poisson_blend
from xiyuan_mvp.clear_fusion import fuse
from xiyuan_mvp.config import load_config
from xiyuan_mvp.image_io import write_image
from xiyuan_mvp.pipeline import StitchPipeline
import xiyuan_mvp.pipeline as pipeline_module
from xiyuan_mvp.run_io import environment, save_run, sha256, write_json
from xiyuan_mvp.seam_mask import mask_from_binary
from xiyuan_mvp.types import RegistrationResult, StitchResult


def joint_cases(root):
    cases = []
    for name in ('brick', 'grass', 'gravel', 'coffee'):
        raw = getattr(data, name)()
        ref = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR if raw.ndim == 2 else cv2.COLOR_RGB2BGR)
        h, w = ref.shape[:2]
        start, cut = int(w * .3), int(w * .7)
        for lighting in (False, True):
            a, b = ref[:, :cut].copy(), ref[:, start:].copy()
            for im, offset, direction in ((a, 0, 1), (b, start, -1)):
                yy, xx = np.indices(im.shape[:2], dtype=np.float32)
                displacement = 4 * np.exp(-((xx + offset - (start+cut)/2)/(w*.12))**2) * np.sin(yy/(h/8)+.37)
                im[:] = cv2.remap(im, xx, yy+direction*displacement, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            if lighting:
                b[:] = np.clip(b.astype(float)*np.linspace(.85, 1.15, h)[:, None, None], 0, 255).astype(np.uint8)
            ca, cb = np.zeros_like(ref), np.zeros_like(ref)
            ma, mb = np.zeros((h, w), np.uint8), np.zeros((h, w), np.uint8)
            ca[:, :cut], cb[:, start:] = a, b
            ma[:, :cut], mb[:, start:] = 255, 255
            cid = f'{name}_joint_' + ('light' if lighting else 'warp')
            folder = root/cid
            for key, im in (('a', a), ('b', b), ('reference', ref)):
                write_image(folder/(key+'.png'), im)
            case = {'id': cid, 'source_id': name, 'kind': 'derived_photo_pair',
                    'suite': 'dual_damage_stress', 'split': 'development',
                    'image_a': str(folder/'a.png'), 'image_b': str(folder/'b.png'),
                    'reference': str(folder/'reference.png'), 'reference_to_a': np.eye(3).tolist(),
                    'note': 'Known crop frame, opposite local warps; isolates repair, not registration accuracy.'}
            reg = RegistrationResult(a, b, ca, cb, ma, mb, ma & mb,
                np.array([[1, 0, start], [0, 1, 0], [0, 0, 1]], float), np.eye(3), np.zeros((8, 8, 3), np.uint8),
                {'feature_method': 'known_crop_transform'})
            traditional = feather_blend(reg)
            mask = mask_from_binary(reg.overlap_mask, reg.overlap_mask, 12)
            prepared = StitchResult(traditional, traditional.copy(), reg, mask,
                {'use_ai': False}, poisson_blend(reg), reg.overlap_mask, [], traditional)
            cases.append((case, prepared))
    return cases


def run():
    assert torch.cuda.is_available()
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1., 8*1024**3/total))
    cfg = load_config()
    cfg['registration'].update(method='loftr', loftr_device='cuda', loftr_max_size=640)
    cfg['neural_alignment'].update(device='cuda', preserve_detail=True, method='raft_large')
    cfg['inpainting'].update(engine='hybrid', device='cuda', local_files_only=False,
        cache_dir='/kaggle/temp/xiyuan-hf-cache', controlnet='canny', controlnet_scale=.8,
        controlnet_mask_mode='none', patch_layout='single', patch_size=512,
        steps=20, strength=.25, guidance_scale=3.5, use_lcm=False,
        hybrid_diffusion_policy='on_structural_gain', hybrid_min_residual_error=2.)
    cfg['blend'].update(ai_opacity=.05, ai_boundary_guard=False)
    cfg['clear_fusion'].update(enabled=True, clarity_margin=1.1)
    started = time.perf_counter()
    rows = []
    report = {'status': 'running', 'experiment': 'paired-stage-ablation-v12', 'records': rows,
              'gpu': torch.cuda.get_device_name(0), 'environment': environment(),
              'allocator_limit_bytes': min(total, 8*1024**3),
              'protocol': {'margins': [1.1, 1.05], 'config': deepcopy(cfg),
                 'references_used_by_repair': False, 'split': 'development regression and dual-damage stress',
                 'same_reference_masks_per_case': True, 'same_diffusion_pixels_for_feather_replay': True,
                 'guard_enabled': False, 'opacity': .05},
              'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in
                  list((ROOT/'xiyuan_mvp').glob('*.py')) + [ROOT/'xiyuan_mvp/default.yaml',
                  ROOT/'scripts/hybrid_ablation_research_gpu.py', ROOT/'scripts/refined_research_gpu.py']}}
    original_guard = pipeline_module.apply_ai_guard
    captured = {}
    def observe_composition(base, candidate, binary, soft, blend):
        # Observe genuine model output; inference and all runtime decisions stay unchanged.
        captured['candidate'] = candidate.copy()
        return apply_ai_guard(base, candidate, binary, soft, blend)
    try:
        cases = [(dict(c, suite='historical_30'), None) for c in inputs() + create_cases(DATA/'deformation')]
        cases += joint_cases(DATA/'joint-ablation')
        for case, _ in cases:
            case['input_sha256'] = {k: sha256(case['image_'+k]) for k in ('a', 'b')}
        write_json(OUTPUT/'inputs-ablation-v12.json', {'cases': [c for c, _ in cases], **report['protocol']})
        # This file is written before looking at any quality scores.
        write_json(OUTPUT/'protocol-ablation-v12.json', {k: v for k, v in report.items() if k != 'records'})
        ev = LPIPSEvaluator()
        pipe = StitchPipeline(deepcopy(cfg))
        pipeline_module.apply_ai_guard = observe_composition
        for case, prepared in cases:
            try:
                registered = prepared or pipe.run(case['image_a'], case['image_b'], use_ai=False)
                save_run(OUTPUT/'registered'/case['id'], registered, pipe.config, extra={'case': case})
            except Exception as exc:
                rows.append({'id': case['id'], 'suite': case['suite'], 'status': 'failed',
                             'stage': 'registration', 'error': str(exc)})
                write_json(OUTPUT/'progress-ablation-v12.json', {'records': rows})
                continue
            for margin in (1.1, 1.05):
                row = {'id': case['id'], 'suite': case['suite'], 'margin': margin, 'status': 'failed'}
                folder = OUTPUT/f'margin-{margin:.2f}'/case['id']
                captured.clear()
                try:
                    pipe.config['clear_fusion']['clarity_margin'] = margin
                    clear, cm = fuse(registered.registration, pipe.config['clear_fusion'])
                    base = replace(registered, final_image=clear, repair_base_image=clear,
                        metrics={**{k: v for k, v in registered.metrics.items() if not k.startswith('clear_')},
                                 **{'clear_'+k: v for k, v in cm.items()}, 'clear_fusion_enabled': True})
                    result = pipe.run(case['image_a'], case['image_b'], prepared=base, use_ai=True)
                    raft = result.repair_base_image
                    outside = result.mask.soft_mask == 0
                    assert np.array_equal(result.final_image[outside], clear[outside])
                    assert np.array_equal(raft[outside], clear[outside])
                    images = {'clear': clear, 'raft': raft, 'hybrid': result.final_image}
                    if result.metrics['diffusion_executed']:
                        candidate = captured['candidate']
                        weight = (result.mask.soft_mask.astype(np.float32)/255*.05)[..., None]
                        old = np.clip(np.rint(raft.astype(np.float32)*(1-weight)+candidate.astype(np.float32)*weight), 0, 255).astype(np.uint8)
                        old[outside] = raft[outside]
                        images['legacy_double_feather'] = old
                        write_image(folder/'diffusion_candidate.png', candidate)
                    else:
                        assert np.array_equal(result.final_image, raft)
                        images['legacy_double_feather'] = raft
                    scores = quality(images, case, result, ev)
                    for name, im in images.items():
                        write_image(folder/(name+'.png'), im)
                    save_run(folder/'run', result, pipe.config, extra={'case': case})
                    row.update(status='success', quality=scores, metrics=result.metrics,
                               outside_mask_unchanged=True)
                except Exception as exc:
                    row['error'] = f'{type(exc).__name__}: {exc}'
                    traceback.print_exc()
                rows.append(row)
                write_json(folder/'evaluation.json', row)
                write_json(OUTPUT/'progress-ablation-v12.json', {'records': rows})
                print(case['id'], margin, row['status'], flush=True)
        report['status'] = 'completed' if all(r['status'] == 'success' for r in rows) else 'completed_with_failures'
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        pipeline_module.apply_ai_guard = original_guard
        report['elapsed_seconds'] = time.perf_counter()-started
        write_json(OUTPUT/'report-ablation-v12.json', report)
        with zipfile.ZipFile('/kaggle/working/ablation-v12-bundle.zip', 'w', zipfile.ZIP_DEFLATED) as z:
            for p in OUTPUT.rglob('*'):
                if p.is_file(): z.write(p, p.relative_to(OUTPUT.parent))


if __name__ == '__main__':
    run()
