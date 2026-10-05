"""Summarize paired stages without crediting RAFT's improvements to diffusion."""
import argparse
import json
from pathlib import Path
import statistics

from xiyuan_mvp.run_io import write_json

METRICS = ('mae', 'lpips', 'ssim')
STAGES = ('clear', 'raft', 'hybrid', 'legacy_double_feather')


def scores(rows, stage, region='seam'):
    values = [r['quality'][stage][region] for r in rows if r.get('quality')]
    return {k: statistics.mean(v[k] for v in values) for k in METRICS} if values else None


def compare(rows, before, after, region='seam'):
    deltas = []
    for row in rows:
        if not row.get('quality'):
            continue
        q = row['quality']
        deltas.append({'id': row['id'], **{k: q[after][region][k]-q[before][region][k] for k in METRICS}})
    by_metric = {}
    for key in METRICS:
        direction = -1 if key == 'ssim' else 1
        by_metric[key] = {'mean_delta': statistics.mean(d[key] for d in deltas) if deltas else None,
            'improved': sum(d[key]*direction < -1e-8 for d in deltas),
            'worse': sum(d[key]*direction > 1e-8 for d in deltas),
            'tied': sum(abs(d[key]) <= 1e-8 for d in deltas)}
    return {'paired_references': len(deltas), 'metrics': by_metric, 'cases': deltas}


def summarize(report):
    rows = report['records']
    successful = [r for r in rows if r['status'] == 'success']
    keys = [(r['id'], r['margin']) for r in successful]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate case/margin records')
    result = {'status': report.get('status'), 'failures': [r for r in rows if r['status'] != 'success'],
              'groups': [], 'margin_comparisons': []}
    for suite in sorted({r['suite'] for r in successful}):
        for margin in sorted({r['margin'] for r in successful}):
            subset = [r for r in successful if r['suite'] == suite and r['margin'] == margin]
            generated = [r for r in subset if r['metrics']['diffusion_executed']]
            refs = [r for r in subset if r.get('quality')]
            stats = {'suite': suite, 'margin': margin, 'successful': len(subset), 'references': len(refs),
                     'no_reference': len(subset)-len(refs), 'generated': len(generated),
                     'raft_executed': sum(r['metrics'].get('model_executed', False) for r in subset),
                     'quality': {s: scores(subset, s) for s in STAGES},
                     'generated_quality': {s: scores(generated, s) for s in STAGES},
                     'hybrid_vs_raft': compare(subset, 'raft', 'hybrid'),
                     'hybrid_vs_clear': compare(subset, 'clear', 'hybrid'),
                     'single_vs_double_feather': compare(generated, 'legacy_double_feather', 'hybrid')}
            for key in ('ai_seconds', 'neural_repair_seconds', 'inference_seconds', 'peak_vram_mb'):
                values = [r['metrics'][key] for r in subset if r['metrics'].get(key) is not None]
                stats[key] = {'median': statistics.median(values), 'max': max(values)} if values else None
            result['groups'].append(stats)
        old = {r['id']: r for r in successful if r['suite'] == suite and r['margin'] == 1.1}
        new = {r['id']: r for r in successful if r['suite'] == suite and r['margin'] == 1.05}
        paired = []
        for cid in sorted(old.keys() & new.keys()):
            if old[cid].get('quality') and new[cid].get('quality'):
                paired.append({'id': cid, 'quality': {'old': old[cid]['quality']['hybrid'],
                                                     'new': new[cid]['quality']['hybrid']}})
        result['margin_comparisons'].append({'suite': suite, '1.05_minus_1.10': compare(paired, 'old', 'new')})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('report', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(json.loads(args.report.read_text(encoding='utf-8')))
    write_json(args.output, summary)
    for group in summary['groups']:
        print(group['suite'], group['margin'], 'success', group['successful'], 'generated', group['generated'],
              'hybrid LPIPS delta vs RAFT', group['hybrid_vs_raft']['metrics']['lpips'])
