"""Offline seam diagnostics. Alerts never certify improvement or alter acceptance."""
from collections import Counter
from datetime import datetime
from pathlib import Path
import json

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

from .core import file_hash, write_json
from .hybrid_experiment import BASE, read


def validate_arrays(base, candidate, band):
    if base.dtype != np.uint8 or candidate.dtype != np.uint8:
        raise ValueError('Expected uint8 RGB images')
    if base.ndim != 3 or base.shape[2] != 3 or candidate.shape != base.shape:
        raise ValueError('Images must share the exact RGB canvas')
    if band.dtype != bool or band.shape != base.shape[:2] or not band.any():
        raise ValueError('Expected a nonempty boolean editable mask')


def gradient_edges(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy) > 60


def diagnostics(base, candidate, band):
    validate_arrays(base, candidate, band)
    count = int(band.sum())
    black = band & (base.max(axis=2) > 40) & (candidate.max(axis=2) < 8)
    white = band & (base.max(axis=2) < 215) & (candidate.min(axis=2) > 247)
    # Ignore the edit border and allow a 2-pixel edge displacement.
    interior = cv2.erode(band.astype(np.uint8), np.ones((7, 7), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    old_edges = gradient_edges(base) & interior
    new_nearby = cv2.dilate(gradient_edges(candidate).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    edge_count = int(old_edges.sum())
    lost_fraction = float((old_edges & ~new_nearby).sum() / edge_count) if edge_count >= 64 else None
    outside_changes = int(np.any(base[~band] != candidate[~band], axis=1).sum())
    metrics = {'new_black_fraction': float(black.sum() / count), 'new_white_fraction': float(white.sum() / count),
               'baseline_edge_pixels': edge_count, 'missing_edge_fraction': lost_fraction,
               'protected_pixel_changes': outside_changes}
    alerts = []
    if outside_changes:
        alerts.append({'code': 'protected_change', 'message': f'保护区有 {outside_changes} 个像素变化。'})
    if metrics['new_black_fraction'] > .05:
        alerts.append({'code': 'new_black', 'message': f'编辑区 {metrics["new_black_fraction"]:.1%} 变成新增近黑像素。'})
    if metrics['new_white_fraction'] > .05:
        alerts.append({'code': 'new_white', 'message': f'编辑区 {metrics["new_white_fraction"]:.1%} 变成新增近白像素，需检查是否过曝或缺失。'})
    if lost_fraction is not None and lost_fraction > .60:
        alerts.append({'code': 'edge_loss', 'message': f'原有明显边缘中 {lost_fraction:.1%} 在附近未找到，需检查是否抹平纹理。'})
    return {'status': 'risk' if alerts else 'review_needed', 'metrics': metrics, 'alerts': alerts,
            'label': '发现退化风险，需复核' if alerts else '未发现已覆盖的异常，仍需人工判断',
            'rules': {'new_black_or_white_fraction': .05, 'edge_loss_fraction': .60, 'minimum_edge_pixels': 64, 'edge_tolerance_pixels': 2},
            'scope': 'No reference used. May flag intended changes and miss wrong text, changed objects or geometry. No automatic adoption decision.'}


def reference_comparison(base, candidate, reference, band):
    validate_arrays(base, candidate, band)
    if reference.shape != base.shape or reference.dtype != np.uint8:
        raise ValueError('Reference must share the exact canvas; no implicit resize or alignment')
    results = {}
    for name, picture in [('traditional', base), ('candidate', candidate)]:
        score, similarity = structural_similarity(reference, picture, channel_axis=2, data_range=255, full=True)
        delta = np.abs(picture.astype(np.float32) - reference.astype(np.float32))
        results[name] = {'global_ssim': float(score), 'seam_ssim': float(similarity.mean(axis=2)[band].mean()),
                         'seam_mae': float(delta[band].mean())}
    s = results['candidate']['seam_ssim'] - results['traditional']['seam_ssim']
    e = results['candidate']['seam_mae'] - results['traditional']['seam_mae']
    if abs(s) < 1e-6 and abs(e) < 1e-6:
        verdict, label = 'unchanged', '参考图局部指标无变化'
    elif s < -1e-6 and e > 1e-6:
        verdict, label = 'worse', '参考图显示局部指标退步'
    elif s > 1e-6 and e < -1e-6:
        verdict, label = 'better', '参考图显示局部指标改善'
    else:
        verdict, label = 'mixed', '参考图局部指标结论不一致'
    return {'available': True, 'verdict': verdict, 'label': label, **results,
            'seam_ssim_change': s, 'seam_mae_change': e,
            'scope': 'Same-canvas controlled references only. SSIM uses 7x7 windows, averaged over editable pixels; boundary windows include neighbors. No significance test or semantic guarantee.'}


def detail_crop(base, candidate, band):
    """Same crop for both images, centered on greatest aggregate pixel change."""
    height, width = band.shape
    size = min(max(96, round(min(height, width) * .28)), min(height, width), 320)
    difference = np.abs(candidate.astype(np.float32) - base.astype(np.float32)).mean(axis=2) * band
    density = cv2.boxFilter(difference, -1, (size, size), normalize=False, borderType=cv2.BORDER_CONSTANT)
    density[~band] = -1
    y, x = np.unravel_index(np.argmax(density), density.shape)
    x = int(np.clip(x - size // 2, 0, width - size))
    y = int(np.clip(y - size // 2, 0, height - size))
    return [x, y, x + size, y + size]


def audit_case(folder, case):
    cell = folder / case['folder']
    names = ['traditional.png', 'hybrid.png', 'safe.png', 'raw.png', 'editable.png']
    before = {name: file_hash(cell / name) for name in names}
    base = np.array(Image.open(cell / 'traditional.png').convert('RGB'))
    candidate = np.array(Image.open(cell / 'hybrid.png').convert('RGB'))
    band = np.array(Image.open(cell / 'editable.png')) > 0
    result = {'version': 'seam-diagnostics-v1', 'number': case['number'], 'evaluated_file': 'hybrid.png',
              'diagnostics': diagnostics(base, candidate, band), 'input_sha256': before,
              'reference': {'available': False, 'label': '没有完整参考图，无法计算局部保真分数'},
              'original_acceptance': read(cell / 'acceptance.json')['status']}
    crop = detail_crop(base, candidate, band)
    Image.fromarray(base).crop(crop).save(cell / 'audit-base.png')
    Image.fromarray(candidate).crop(crop).save(cell / 'audit-candidate.png')
    if case['reference']:
        reference = np.array(Image.open(cell / 'reference.png').convert('RGB'))
        result['reference'] = reference_comparison(base, candidate, reference, band)
        Image.fromarray(reference).crop(crop).save(cell / 'audit-reference.png')
        result['input_sha256']['reference.png'] = file_hash(cell / 'reference.png')
    delta = np.abs(candidate.astype(np.float32) - base.astype(np.float32)).mean(axis=2)
    alpha = (np.clip(delta / 48, 0, .8) * band)[..., None]
    overlay = np.rint(base * (1 - alpha) + np.array([255, 40, 40]) * alpha).astype(np.uint8)
    Image.fromarray(overlay).save(cell / 'audit-change-map.png')
    result['crop_xyxy'] = crop
    result['crop_note'] = '两图使用完全相同的裁剪位置，选取累计像素变化最多的邻域；变化最多不等于错误最多。'
    result['original_files_unchanged'] = all(file_hash(cell / name) == before[name] for name in names)
    if not result['original_files_unchanged']:
        raise RuntimeError('Source evidence changed during audit')
    write_json(cell / 'quality-audit.json', result)
    return result


def audit(folder):
    spec = read(folder / 'experiment.json')
    rows = [audit_case(folder, c) for c in spec['cases']]
    summary = {'version': 'seam-diagnostics-v1', 'created_at': datetime.now().astimezone().isoformat(),
               'new_model_calls': 0, 'cases': rows, 'no_reference_alert_counts': dict(Counter(a['code'] for c in rows for a in c['diagnostics']['alerts'])),
               'note': 'Post-hoc diagnostics on six existing pilot results. Thresholds are exploratory, not independently validated. Does not overwrite candidate, final output, or prior acceptance.'}
    write_json(folder / 'quality-audit.json', summary)
    return summary


if __name__ == '__main__':
    folder = Path(read(BASE / 'hybrid-latest.json')['folder'])
    result = audit(folder)
    print(json.dumps([{'number': c['number'], 'alerts': [a['code'] for a in c['diagnostics']['alerts']], 'metrics': c['diagnostics']['metrics'],
                       'reference': c['reference'].get('verdict')} for c in result['cases']], ensure_ascii=False))
