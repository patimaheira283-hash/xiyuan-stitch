"""Audit saved pilot artifacts offline; never generate or edit an image."""
from datetime import datetime
from pathlib import Path
import json

import numpy as np
from PIL import Image

from .core import artifact_root, file_hash, write_json
from .hybrid_experiment import BASE, read, check_new_black_regions


def validate(folder):
    spec = read(folder / 'experiment.json')
    original = read(BASE / 'experiment.json')
    source_cases = {c['number']: c for c in original['cases']}
    gallery = {c['number']: c for c in read(folder / 'data.json')['cases']}
    stats = read(folder / 'statistics.json')
    stats_cases = {c['number']: c for c in stats['cases']}
    root = artifact_root().resolve()
    records = []
    for c in spec['cases']:
        cell = folder / c['folder']
        source_case = source_cases[c['number']]
        source = BASE / source_case['folder']
        generation = read(cell / 'generation.json')
        meta = read(cell / 'hybrid.json')
        acceptance = read(cell / 'acceptance.json')
        checks = {}
        checks['frozen_input_hashes'] = all(file_hash(cell / name) == digest for name, digest in c['input_sha256'].items())
        originals = ['left.png', 'right.png', 'traditional.png', 'gpt.png', 'traditional-mask.png']
        checks['first_round_files_unchanged'] = all(file_hash(cell / name) == file_hash(source / name) for name in originals)
        if c['reference']:
            checks['reference_unchanged'] = file_hash(cell / 'reference.png') == file_hash(source / source_case['reference'])
        artifact = Path(generation['artifact_path'])
        checks['original_is_under_current_imagegen_root'] = artifact.is_absolute() and artifact.resolve().is_relative_to(root)
        raw_hash = file_hash(cell / 'raw.png')
        checks['original_artifact_hash'] = artifact.is_file() and file_hash(artifact) == raw_hash == generation['output_sha256']
        base = np.array(Image.open(cell / 'traditional.png').convert('RGB'))
        candidate = np.array(Image.open(cell / 'hybrid.png').convert('RGB'))
        final = np.array(Image.open(cell / 'safe.png').convert('RGB'))
        band = np.array(Image.open(cell / 'editable.png')) > 0
        changes = int(np.any(candidate[~band] != base[~band], axis=1).sum())
        checks['zero_protected_pixel_changes'] = changes == 0 == meta['protected_pixel_changes']
        checks['final_protected_pixels'] = bool(np.array_equal(final[~band], base[~band]))
        checks['native_dimensions'] = list(Image.open(cell / 'hybrid.png').size) == c['base_size'] == list(Image.open(cell / 'safe.png').size)
        checks['requested_canvas_dimensions'] = list(Image.open(cell / 'raw.png').size) == c['canvas_size'] == list(Image.open(cell / 'canvas.png').size)
        with Image.open(cell / 'mask.png') as mask:
            checks['alpha_mask_dimensions'] = mask.mode == 'RGBA' and list(mask.size) == c['canvas_size']
        checks['retained_candidate_hash'] = file_hash(cell / 'hybrid.png') == meta['output_sha256']
        expected = cell / ('hybrid.png' if acceptance['accepted'] else 'traditional.png')
        checks['final_file_is_exact_accepted_or_fallback'] = file_hash(cell / 'safe.png') == file_hash(expected) == acceptance['output_sha256']
        black = check_new_black_regions(base, candidate, band)
        checks['recorded_gate_matches_pixels'] = black == acceptance['black_region_check'] and acceptance['accepted'] == (changes == 0 and black['passed'])
        if not acceptance['accepted']:
            checks['failed_candidate_not_overwritten'] = file_hash(cell / 'hybrid.png') != file_hash(cell / 'safe.png')
        row = gallery[c['number']]
        checks['gallery_resources_exist'] = all((cell / name).is_file() for name in row['files'])
        checks['gallery_required_images'] = all(name in row['files'] for name in ['left.png', 'right.png', 'traditional.png', 'gpt.png', 'safe.png', 'hybrid.png', 'raw.png', 'mask-guide.png', 'aligned-candidate.png'])
        checks['gallery_final_notes'] = row['review'] == read(folder / 'reviews.json')[str(c['number'])]['review'] and row['acceptance']['status'] == acceptance['status']
        checks['statistics_match_record'] = stats_cases[c['number']]['accepted'] == acceptance['accepted'] and stats_cases[c['number']]['generation_seconds'] == generation['elapsed_seconds']
        records.append({'number': c['number'], 'checks': checks, 'passed': all(checks.values()), 'protected_pixel_changes': changes,
                        'accepted': acceptance['accepted'], 'original_artifact': artifact.as_posix(), 'original_sha256': raw_hash})
    top = {
        'six_fixed_cases': [c['number'] for c in spec['cases']] == [1, 4, 9, 15, 22, 23],
        'six_successful_images': all(read(folder / c['folder'] / 'generation.json')['status'] == 'succeeded' for c in spec['cases']),
        'five_accepted_one_fallback': sum(c['accepted'] for c in records) == 5,
        'initial_400_retained': 'HTTP 400' in read(folder / '01/pilot-with-tool-limit-generation.json')['error'],
        'report_and_gallery_present': all((folder / name).is_file() for name in ['report.md', 'index.html', 'data.json']),
        'parent_gallery_link': folder.name + '/index.html' in (BASE / 'index.html').read_text(encoding='utf-8'),
    }
    result = {'checked_at': datetime.now().astimezone().isoformat(), 'checks': top, 'cases': records,
              'passed': all(top.values()) and all(c['passed'] for c in records),
              'scope': 'File integrity, frozen inputs, exact protected pixels, dimensions, recorded gate decisions and gallery resources; not a semantic accuracy score.'}
    write_json(folder / 'validation.json', result)
    return result


if __name__ == '__main__':
    folder = Path(read(BASE / 'hybrid-latest.json')['folder'])
    result = validate(folder)
    print(json.dumps({'passed': result['passed'], 'cases': len(result['cases']), 'check_count': len(result['checks']) + sum(len(c['checks']) for c in result['cases']),
                      'failed': [name for name, passed in result['checks'].items() if not passed] + [f"{c['number']}: {name}" for c in result['cases'] for name, passed in c['checks'].items() if not passed]}))
    raise SystemExit(0 if result['passed'] else 1)
