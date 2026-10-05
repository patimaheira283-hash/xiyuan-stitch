import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]


def summarize(path):
    report=json.loads(Path(path).read_text(encoding='utf-8'))
    groups={}
    for r in report['records']:
        if r.get('status')!='success':
            print('FAIL',r.get('id'),r.get('profile'),r.get('error'));continue
        for name, regions in (r.get('quality') or {}).items():
            for region,metrics in regions.items():
                if isinstance(metrics,dict):
                    groups.setdefault((name,region),[]).append(metrics['lpips'])
    for key,values in groups.items():print(key,'n=',len(values),'mean=',round(statistics.mean(values),6))
    for profile in sorted({r.get('profile','') for r in report['records']}):
        rows=[r for r in report['records'] if r.get('profile')==profile and r.get('status')=='success']
        times=[r['metrics']['inference_seconds'] for r in rows if 'inference_seconds' in r.get('metrics',{})]
        print('PROFILE',profile,'success=',len(rows),'inference median=',statistics.median(times) if times else None)
    print('STATUS',report['status'],'seconds=',report['elapsed_seconds'])


if __name__=='__main__':
    import sys
    summarize(sys.argv[1])
