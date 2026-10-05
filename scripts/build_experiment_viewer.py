"""Assemble an offline comparison workbench from saved, unmodified GPU outputs."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
from xiyuan_mvp.run_io import write_json

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT/'deliverables/experiments'
LABELS = {'Feather':'羽化', 'Poisson':'泊松', 'exposure_only':'曝光校正',
          'dis':'传统 DIS', 'raft_large':'RAFT Large', 'raft_refined':'RAFT＋DIS',
          'raft_local':'RAFT＋局部光照', 'hybrid_canny':'RAFT＋Canny 生成',
          'hybrid_tile_lcm':'RAFT＋Tile LCM 生成'}


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    experiments = {}; flat = []
    for version in ('v4','v5'):
        root = ROOT/'outputs/research'/version
        report_path = root/'research-results'/f'report-{version}.json'
        if not report_path.exists(): continue
        report = json.loads(report_path.read_text(encoding='utf-8'))
        raw = DEST/'raw'/version; raw.mkdir(parents=True, exist_ok=True)
        shutil.copy2(report_path, raw/report_path.name)
        snapshot = ROOT/'outputs/research'/f'submitted-{version}'
        shutil.copytree(snapshot,raw/'submitted',dirs_exist_ok=True)
        cases = {}
        rows = report['records']
        ids = [r['id'] for r in rows if r.get('profile')=='baseline' and
               any(r['id'].startswith(s) for s in ('fine_grained_wood_','dark_wooden_planks_','brick_wall_003_','brick_wall_005_'))]
        with zipfile.ZipFile(root/f'research-{version}-bundle.zip') as z:
            names = set(z.namelist())
            def copy_artifact(member, target):
                if member not in names: return None
                p=DEST/target; p.parent.mkdir(parents=True,exist_ok=True)
                p.write_bytes(z.read(member)); return target
            for cid in ids:
                baseline = next(r for r in rows if r['id']==cid and r['profile']=='baseline')
                if baseline['status']!='success': continue
                prefix = f'research-results/baseline/{cid}/'
                run = json.loads(z.read(prefix+'run.json'))
                entry = {'id':cid, 'bbox':run['mask_bbox'], 'methods':{},
                         'category':'木纹' if 'wood' in cid else '砖墙', 'regression':False}
                for key in ('input_a','input_b','05_seam_mask'):
                    name=run['artifacts'][key]['file']
                    entry[key]=copy_artifact(prefix+name,f'images/{version}/{cid}/{name}')
                for method,artifact in (('Feather','06_traditional'),('Poisson','06_poisson')):
                    file=run['artifacts'][artifact]['file']
                    metrics=baseline['metrics']
                    seconds=metrics['feather_seconds' if method=='Feather' else 'poisson_seconds']
                    entry['methods'][method]={'image':copy_artifact(prefix+file,f'images/{version}/{cid}/{file}'),
                        'quality':baseline['quality'][method], 'seconds':seconds, 'vram':None, 'error':None}
                for row in rows:
                    if row['id']!=cid or row['profile']=='baseline': continue
                    method=row['profile']; metrics=row.get('metrics',{})
                    prefix2=f'research-results/{method}/{cid}/'
                    name='07_ai_result.png' if method.startswith('hybrid') else 'result.png'
                    entry['methods'][method]={'image':copy_artifact(prefix2+name,f'images/{version}/{cid}/{method}.png'),
                        'quality':(row.get('quality') or {}).get(method),
                        'seconds':metrics.get('ai_seconds',metrics.get('repair_seconds')),
                        'vram':metrics.get('peak_vram_mb'), 'error':row.get('error')}
                base_lpips=entry['methods']['Feather']['quality']['seam']['lpips']
                raft=entry['methods'].get('raft_large',{}).get('quality')
                entry['regression']=bool(raft and raft['seam']['lpips']>base_lpips)
                for method,info in entry['methods'].items():
                    for region,q in (info['quality'] or {}).items():
                        flat.append({'experiment':version,'case':cid,'method':method,'region':region,
                            'mae':q.get('mae'),'ssim':q.get('ssim'),'lpips':q.get('lpips'),
                            'seconds':info['seconds'],'peak_vram_mb':info['vram'],'error':info['error']})
                cases[cid]=entry
        experiments[version]={'cases':cases, 'gpu':report.get('gpu'),
            'total_records':len(rows),'runtime_failures':sum(r['status']!='success' for r in rows),
            'seconds':report.get('elapsed_seconds'),'raw':f'raw/{version}/{report_path.name}'}
    data={'experiments':experiments,'labels':LABELS}
    write_json(DEST/'data.json',data)
    with (DEST/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    template=(ROOT/'scripts/experiment_viewer.html').read_text(encoding='utf-8')
    (DEST/'index.html').write_text(template.replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('</','<\\/')),encoding='utf-8')
    shutil.copy2(ROOT/'data/neural-validation/sources.json',DEST/'sources.json')
    # Engineering acceptance and cloud desktop round trip are separate from quality scores.
    for path,name in ((ROOT/'outputs/portable-workflow-030/acceptance.json','desktop-acceptance.json'),
                      (ROOT/'outputs/high-resolution-acceptance/acceptance.json','4k-acceptance.json'),
                      (ROOT/'outputs/desktop-acceptance/roundtrip-check.json','cloud-roundtrip.json')):
        if path.exists(): shutil.copy2(path,DEST/name)
    print(json.dumps({'cases':{v:len(e['cases']) for v,e in experiments.items()},'metric_rows':len(flat)}))


if __name__=='__main__':main()
