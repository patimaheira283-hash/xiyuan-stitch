"""Package the tested 0.4 artifacts and check every archived byte."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil
import subprocess
import zipfile

from scripts.package_delivery import package

ROOT = Path(__file__).resolve().parents[1]
VERSION = '0.4.0'
OUT = ROOT / 'deliverables'


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def archive_tree(folder, target):
    paths = sorted(p for p in folder.rglob('*') if p.is_file())
    manifest = {p.relative_to(folder.parent).as_posix(): digest(p) for p in paths}
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in paths:
            z.write(p, p.relative_to(folder.parent).as_posix())
    with zipfile.ZipFile(target) as z:
        assert set(z.namelist()) == set(manifest)
        for name, expected in manifest.items():
            with z.open(name) as f:
                assert hashlib.file_digest(f, 'sha256').hexdigest() == expected, name
    return manifest


def main():
    workflow = json.loads((ROOT/'outputs/portable-workflow-040/acceptance.json').read_text())
    assert workflow['status'] == 'passed' and workflow['frozen'] and workflow['network_blocked']
    assert len(workflow['records']) == 3
    for row in workflow['records']:
        assert row['status'] == 'passed'
        assert row['metrics']['clear_algorithm_revision'] == 'adaptive-noise-v3'
        assert row['metrics']['repair_engine'] == 'raft_large'
    high = json.loads((ROOT/'outputs/high-resolution-040/acceptance.json').read_text())
    assert high['status'] == 'passed' and high['output_shape'] == [1802, 4096, 3]
    assert '36 passed' in (ROOT/'.work/tests-final-040.log').read_text()

    external = Path('D:/xiyuan-delivery-check-20260910')
    installed = json.loads(subprocess.check_output([
        str(external/'.venv/Scripts/python.exe'), '-c',
        "import xiyuan_mvp,json,importlib.metadata; from pathlib import Path; import hashlib; "
        "p=Path(xiyuan_mvp.__file__).parent; "
        "print(json.dumps({'version':importlib.metadata.version('xiyuan-stitch'), 'module':str(p), "
        "'clear_sha256':hashlib.sha256((p/'clear_fusion.py').read_bytes()).hexdigest()}))"
    ], cwd=external, text=True))
    assert installed['version'] == VERSION and 'site-packages' in installed['module']
    assert installed['clear_sha256'] == digest(ROOT/'xiyuan_mvp/clear_fusion.py')
    run = json.loads((external/'output-clear-040-verified/run.json').read_text())
    assert run['status'] == 'success' and run['metrics']['feature_method'] == 'LoFTR'
    assert run['metrics']['clear_algorithm_revision'] == 'adaptive-noise-v3'
    assert run['metrics']['clear_fusion_enabled'] and not run['metrics']['use_ai']
    installed['status'] = 'passed'
    installed['run'] = run
    (ROOT/'outputs/source-install-040.json').write_text(json.dumps(installed, indent=2), encoding='utf-8')

    viewer = OUT/'clear-experiments-0.4.0'
    checks = viewer/'software-checks'
    checks.mkdir(exist_ok=True)
    for src, name in [(ROOT/'outputs/portable-workflow-040/acceptance.json', 'desktop.json'),
                      (ROOT/'outputs/high-resolution-040/acceptance.json', '4k.json'),
                      (ROOT/'outputs/source-install-040.json', 'source-install.json'),
                      (ROOT/'.work/tests-final-040.log', 'tests.txt')]:
        shutil.copy2(src, checks/name)
    for cid in (r['id'] for r in workflow['records']):
        for name in ['clear-desktop.png', 'clear-export.png', 'painted.png', 'desktop.png', 'export.png']:
            dest = checks/cid/name
            dest.parent.mkdir(exist_ok=True)
            shutil.copy2(ROOT/'outputs/portable-workflow-040'/cid/name, dest)

    source_zip = OUT/f'xiyuan-source-{VERSION}.zip'
    package(source_zip)
    with zipfile.ZipFile(source_zip) as z:
        source_manifest = json.loads(z.read('SOURCE_MANIFEST.json'))
        for name, expected in source_manifest.items():
            assert hashlib.sha256(z.read(name)).hexdigest() == expected, name
        assert not any(Path(n).name in ('财务日志.md', '研究报告.md') for n in z.namelist())
    with zipfile.ZipFile(OUT/f'xiyuan_stitch-{VERSION}-py3-none-any.whl') as z:
        for name in z.namelist():
            if name.startswith('xiyuan_mvp/') and not name.endswith('/'):
                assert hashlib.sha256(z.read(name)).hexdigest() == digest(ROOT/name), name
    print('Source archive verified', flush=True)
    desktop = OUT/'0.4.0/XiyuanAI'
    # The tested folder must still contain the binary from the final build.
    assert digest(desktop/'XiyuanAI.exe') == digest(ROOT/'.work/portable-ai/XiyuanAI/XiyuanAI.exe')
    desktop_zip = OUT/f'xiyuan-windows-ai-{VERSION}.zip'
    desktop_manifest = archive_tree(desktop, desktop_zip)
    (OUT/f'windows-files-{VERSION}.json').write_text(json.dumps(desktop_manifest, indent=2), encoding='utf-8')
    print('Desktop archive matches every tested file', flush=True)
    experiments_zip = OUT/f'xiyuan-clear-experiments-{VERSION}.zip'
    archive_tree(viewer, experiments_zip)
    print('Experiment archive verified', flush=True)
    data = json.loads((viewer/'data.json').read_text(encoding='utf-8'))
    quality = {}
    for key, suite in data['suites'].items():
        rows = suite['cases']
        before = [r['methods']['Feather']['quality']['lpips'] for r in rows]
        after = [r['methods']['adaptive']['quality']['lpips'] for r in rows]
        quality[key] = {'count':len(rows), 'feather_lpips':sum(before)/len(rows),
                        'clear_lpips':sum(after)/len(rows),
                        'wins':sum(a<b for a,b in zip(after,before)),
                        'ties':sum(a==b for a,b in zip(after,before)),
                        'losses':sum(a>b for a,b in zip(after,before))}
    artifacts = [source_zip, desktop_zip, experiments_zip, OUT/f'xiyuan_stitch-{VERSION}-py3-none-any.whl']
    manifest = {'version':VERSION, 'created_utc':datetime.now(timezone.utc).isoformat(),
                'artifacts':[{'file':p.name, 'bytes':p.stat().st_size, 'sha256':digest(p)} for p in artifacts],
                'validation':{'core_tests':36, 'offline_packaged_workflows':3,
                              'source_install':'passed; separate directory and site-packages',
                              'all_archive_members_sha256':'passed',
                              'high_resolution_shape':high['output_shape'],
                              'high_resolution_seconds':high['elapsed_seconds']},
                'algorithm_revision':'adaptive-noise-v3', 'quality':quality,
                'quality_scope':'116 controlled regression cases from 16 sources; not untouched holdout. Textures use saved LoFTR geometry; remaining suites use known crops and evaluate fusion only.',
                'entry_points':{'desktop':'0.4.0/XiyuanAI/XiyuanAI.exe',
                                'experiments':'clear-experiments-0.4.0/index.html'}}
    (OUT/f'delivery-manifest-{VERSION}.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
