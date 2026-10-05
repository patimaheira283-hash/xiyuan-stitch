"""Verify the 0.5 delivery against the actually tested installed artifacts."""
from pathlib import Path
import json
import shutil
import hashlib
import zipfile
from scripts.release_clear import archive_tree,digest
from scripts.package_delivery import package

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'deliverables'
VERSION='0.5.0'


def main():
    acceptance=json.loads((ROOT/'outputs/portable-workflow-050/acceptance.json').read_text(encoding='utf-8'))
    assert acceptance['status']=='passed' and acceptance['frozen'] and acceptance['network_blocked']
    assert len(acceptance['records'])==4 and all(r['status']=='passed' for r in acceptance['records'])
    motion=acceptance['records'][-1]
    assert motion['metrics']['model_executed'] and motion['metrics']['repair_applied'] and motion['legacy_option_exercised']
    assert motion['metrics']['repair_revision']=='clarity-guided-raft-v1'
    source_check=json.loads((ROOT/'outputs/source-install-050.json').read_text(encoding='utf-8'))
    assert source_check['version']==VERSION and source_check['model_executed'] and source_check['repair_applied']
    assert source_check['structural_sha256']==digest(ROOT/'xiyuan_mvp/structural_repair.py')
    assert 'site-packages' in source_check['module']
    assert '39 passed' in (ROOT/'.work/tests-050-sequential.log').read_text()
    assert 'passed' in (ROOT/'outputs/source-workflow-050/acceptance.json').read_text()
    viewer=OUT/'structural-experiments-0.5.0';checks=viewer/'software-checks';checks.mkdir(exist_ok=True)
    for p,name in [(ROOT/'outputs/portable-workflow-050/acceptance.json','desktop.json'),
                   (ROOT/'outputs/source-install-050.json','source-install.json'),
                   (ROOT/'.work/tests-050-sequential.log','tests.txt')]:shutil.copy2(p,checks/name)
    for cid in ['fine_grained_wood_local_11','brick_wall_005_local_11','dark_wooden_planks_gradient','protected-motion']:
        shutil.copytree(ROOT/'outputs/portable-workflow-050'/cid,checks/cid,dirs_exist_ok=True)
    logs=viewer/'raw/execution-logs';logs.mkdir(exist_ok=True)
    for name in ['structural-050-regression.log','structural-050-heldout.log','structural-050-real.log',
                 'structural-050-regression-resume.log','structural-050-heldout-resume.log','structural-050-real-resume.log',
                 'structural-050-joint.log','structural-validation-queue.log','tests-050.log']:
        shutil.copy2(ROOT/'.work'/name,logs/name)
    data=json.loads((viewer/'data.json').read_text(encoding='utf-8'));quality={}
    for key,s in data['suites'].items():
        assert len(s['cases'])=={'regression':116,'heldout':64,'real':8,'joint':16}[key]
        q=[c for c in s['cases'] if c['methods']['clear04']['quality']]
        before=[c['methods']['clear04']['quality']['lpips'] for c in q]
        after=[c['methods']['protected']['quality']['lpips'] for c in q]
        quality[key]={'count':len(s['cases']),'scope':s['scope'],
            'clear_lpips':sum(before)/len(before) if before else None,
            'protected_lpips':sum(after)/len(after) if after else None,
            'wins':sum(a<b for a,b in zip(after,before)) if before else None,
            'ties':sum(a==b for a,b in zip(after,before)) if before else None,
            'losses':sum(a>b for a,b in zip(after,before)) if before else None}
    source_zip=OUT/f'xiyuan-source-{VERSION}.zip';package(source_zip)
    with zipfile.ZipFile(source_zip) as z:
        source_manifest=json.loads(z.read('SOURCE_MANIFEST.json'))
        for name,expected in source_manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==expected
    wheel=OUT/f'xiyuan_stitch-{VERSION}-py3-none-any.whl'
    with zipfile.ZipFile(wheel) as z:
        for name in z.namelist():
            if name.startswith('xiyuan_mvp/') and not name.endswith('/'):
                assert hashlib.sha256(z.read(name)).hexdigest()==digest(ROOT/name),name
    desktop=OUT/'0.5.0/XiyuanAI'
    assert digest(desktop/'XiyuanAI.exe')==digest(ROOT/'.work/portable-ai/XiyuanAI/XiyuanAI.exe')
    desktop_zip=OUT/f'xiyuan-windows-ai-{VERSION}.zip'
    manifest=archive_tree(desktop,desktop_zip)
    (OUT/f'windows-files-{VERSION}.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Desktop archive verified byte for byte',flush=True)
    experiments_zip=OUT/f'xiyuan-structural-experiments-{VERSION}.zip';archive_tree(viewer,experiments_zip)
    print('Experiment archive verified byte for byte',flush=True)
    result={'version':VERSION,'artifacts':[{'file':p.name,'bytes':p.stat().st_size,'sha256':digest(p)} for p in [desktop_zip,source_zip,wheel,experiments_zip]],
        'validation':{'core_tests':39,'packaged_workflows':4,'offline':True,'installed_wheel':'passed','archive_hashes':'passed',
        'known_geometry_neural_demo':True,'separate_machine_human_acceptance':False},
        'algorithm_revision':'clarity-guided-raft-v1','quality':quality,
        'execution_issues':'Initial parallel run had two recorded allocation failures and interrupted processes; same frozen algorithm completed sequentially. Original logs and failed attempts retained.',
        'entry_points':{'desktop':'0.5.0/XiyuanAI/XiyuanAI.exe','viewer':'structural-experiments-0.5.0/index.html'}}
    (OUT/f'delivery-manifest-{VERSION}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
