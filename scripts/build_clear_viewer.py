"""Build a portable, offline visual comparison for the 0.4 release."""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import html
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from xiyuan_mvp.image_io import read_image

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'deliverables/clear-experiments-0.4.0'


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    catalog={};flat=[]
    texture_cases={c['id']:c for c in json.loads((ROOT/'data/neural-validation/manifest.json').read_text())['cases']}
    previous=json.loads((ROOT/'deliverables/experiments/data.json').read_text(encoding='utf-8'))['experiments']['v5']['cases']
    suites=[('textures','clear-final-textures','28 个木纹与砖墙案例','同一 LoFTR 配准画布上的完整融合结果；这 28 例来自四张贴图，已用于开发。'),
        ('balanced','clear-final-balanced','24 个左右交替受损案例','四张自然图片，轮流给左图或右图添加扰动；已用于开发。使用已知裁剪坐标，单独检验融合。'),
        ('noise','clear-final-noise','32 个噪声与结构案例','四张图片、四种扰动，含强噪声。初版曾把噪点误当细节；当前为修正后的回归检查。使用已知裁剪坐标。'),
        ('soft','clear-final-soft','32 个弱纹理与低对比案例','四张图片、四种扰动。该素材组曾发现低对比退化，当前为修正后的回归检查。使用已知裁剪坐标。')]
    def copy(p,target):
        out=OUTPUT/target;out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,out);return target
    for key,folder,title,scope in suites:
        src=ROOT/'outputs'/folder
        report=json.loads((src/'results.json').read_text())
        raw=copy(src/'results.json',f'raw/{key}/results.json')
        if (src/'protocol.json').exists():copy(src/'protocol.json',f'raw/{key}/protocol.json')
        cases=[]
        for row in report['records']:
            cid=row['id'];quality=row.get('scores',row.get('quality'))
            item={'id':cid,'images':{},'methods':{},'selection':row['metrics']['adaptive'].get('selection','already aligned')}
            for method,label in [('Feather','羽化融合'),('Poisson','泊松融合'),('adaptive','0.4 清晰融合')]:
                q=quality[method]['full'] if 'full' in quality[method] else quality[method]
                im=copy(src/cid/(method+'.png'),f'images/{key}/{cid}/{method}.png')
                item['methods'][method]={'label':label,'image':im,'quality':q,
                    'seconds':row['metrics']['adaptive']['fusion_seconds'] if method=='adaptive' else None,'device':'CPU'}
            if key=='textures':
                # Verify that the historical RAFT output used the identical Feather canvas.
                old=previous[cid]
                old_feather=ROOT/'deliverables/experiments'/old['methods']['Feather']['image']
                np.testing.assert_array_equal(read_image(old_feather),read_image(src/cid/'Feather.png'))
                raft=old['methods']['raft_large']
                item['methods']['raft03']={'label':'0.3 RAFT','image':copy(ROOT/'deliverables/experiments'/raft['image'],f'images/{key}/{cid}/raft03.png'),
                    'quality':raft['quality']['full'],'seconds':raft['seconds'],'device':'T4 GPU（历史记录）'}
                for name,field in [('a','image_a'),('b','image_b')]:item['images'][name]=copy(Path(texture_cases[cid][field]),f'images/{key}/{cid}/{name}.png')
            else:
                for name in ['a','b','reference']:item['images'][name]=copy(src/cid/(name+'.png'),f'images/{key}/{cid}/{name}.png')
            if key in ['noise','soft']:
                oldroot=ROOT/'outputs'/('clear-adaptive-heldout' if key=='noise' else 'clear-adaptive-v2-fresh')
                oldrows=json.loads((oldroot/'results.json').read_text())['records']
                oldrow=next(r for r in oldrows if r['id']==cid)
                item['methods']['previous']={'label':'修正前的清晰融合','image':copy(oldroot/cid/'adaptive.png',f'images/{key}/{cid}/previous.png'),
                    'quality':oldrow['quality']['adaptive'],'seconds':oldrow['metrics']['adaptive']['fusion_seconds'],'device':'CPU'}
            item['regression']=item['methods']['adaptive']['quality']['lpips']>item['methods']['Feather']['quality']['lpips']
            for method,m in item['methods'].items():flat.append({'suite':key,'case':cid,'method':method,
                **{k:m['quality'][k] for k in ['lpips','ssim','mae']},'seconds':m['seconds'],'device':m['device']})
            cases.append(item)
        catalog[key]={'title':title,'scope':scope,'raw':raw,'cases':cases}
    data={'suites':catalog,'version':'0.4.0','algorithm_sha256':hashlib.sha256((ROOT/'xiyuan_mvp/clear_fusion.py').read_bytes()).hexdigest()}
    # The experiment preceded the addition of a diagnostic revision string.
    # Recover the exact executed module and verify it against the recorded hash.
    executed=(ROOT/'xiyuan_mvp/clear_fusion.py').read_text(encoding='utf-8').replace(",\u0027algorithm_revision\u0027:\u0027adaptive-noise-v3\u0027",'')
    recorded=json.loads((ROOT/'outputs/clear-final-noise/protocol.json').read_text())['algorithm_sha256']
    assert hashlib.sha256(executed.encode()).hexdigest()==recorded,'Experiment source has changed beyond diagnostic metadata'
    snapshot=OUTPUT/'raw/executed-source';snapshot.mkdir(parents=True,exist_ok=True)
    (snapshot/'clear_fusion.py').write_text(executed,encoding='utf-8')
    shutil.copy2(ROOT/'xiyuan_mvp/blending.py',snapshot/'blending.py')
    # Real captures are kept separate from the controlled reference-based scores.
    real_ids=['spw_10_efab8e6e','lpc_library_dd718dea','rew_rew_wall_de776f7d','ges_cave_01_atrium_bad372c8']
    sources={c['id']:c for c in json.loads((ROOT/'data/research-cases.json').read_text())['cases']}
    sections=[]
    for cid in real_ids:
        meta=sources[cid];panels=[]
        base=ROOT/'outputs/research/v4/extracted/research-results/baseline'/cid
        for name,label,p in [('a','输入 A',base/'input_a.png'),('b','输入 B',base/'input_b.png'),
                             ('feather','羽化融合',ROOT/'outputs/clear-final-soft/real'/cid/'Feather.png'),
                             ('clear','0.4 清晰融合',ROOT/'outputs/clear-final-soft/real'/cid/'clear_detail.png')]:
            dest=copy(p,f'images/real/{cid}/{name}.png')
            panels.append(f'<figure><figcaption>{label}</figcaption><a href="{dest}" target="_blank"><img src="{dest}" alt="{label}" loading="lazy"></a></figure>')
        copy(ROOT/'outputs/clear-final-soft/real'/cid/'metrics.json',f'raw/real/{cid}/metrics.json')
        sections.append(f'<section><h2>{html.escape(meta["title"])}</h2><div class="grid">{"".join(panels)}</div><p>来源：{html.escape(meta["repository"])} · {html.escape(meta["revision"][:12])}<br>{html.escape(meta["license"])}<br>{html.escape(meta.get("source_note", ""))}</p></section>')
    (OUTPUT/'real.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>曦源 0.4 · 真实照片</title><style>body{max-width:1480px;margin:24px auto;padding:0 24px;font:15px/1.7 system-ui;background:#f3f5f4;color:#18332e}a{color:#196449}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}figure{margin:0}img{display:block;width:100%;max-height:520px;object-fit:contain;background:#10201b}section{border-top:1px solid #bacac0;margin-top:28px}section p{font-size:12px;color:#536a60}@media(max-width:680px){.grid{grid-template-columns:1fr}}</style><a href="index.html">返回受控案例对比</a><h1>真实照片 · 查看原始结果</h1><p>四组预先选定的实拍图片；没有参考全景，未计算 SSIM 或 LPIPS。两种方法使用同一配准画布。请同时查看整体和局部，留意线条错位、重影、曝光过渡。点击图片打开原尺寸。</p>'''+''.join(sections)+'</html>',encoding='utf-8')
    (OUTPUT/'raw/real/sources.json').write_text(json.dumps([sources[cid] for cid in real_ids],ensure_ascii=False,indent=2),encoding='utf-8')
    preview=Image.new('RGB',(932,712),'#f3f5f4');draw=ImageDraw.Draw(preview)
    font_path=Path('C:/Windows/Fonts/msyh.ttc')
    font=ImageFont.truetype(str(font_path),19) if font_path.exists() else ImageFont.load_default(size=19)
    for col,label in enumerate(['Feather','0.3 RAFT','0.4 Clear']):draw.text((20+col*308,14),label,font=font,fill='#18332e')
    for row,cid in enumerate(['fine_grained_wood_local_11','brick_wall_005_local_11']):
        draw.text((20,54+row*324),cid,font=font,fill='#18332e')
        item=next(c for c in catalog['textures']['cases'] if c['id']==cid)
        for col,key in enumerate(['Feather','raft03','adaptive']):
            im=Image.open(OUTPUT/item['methods'][key]['image']).convert('RGB')
            x,y=(im.width-288)//2,(im.height-288)//2
            preview.paste(im.crop((x,y,x+288,y+288)),(20+col*308,82+row*324))
    preview.save(OUTPUT/'comparison.png')
    (OUTPUT/'data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    template=(ROOT/'scripts/clear_viewer.html').read_text(encoding='utf-8')
    (OUTPUT/'index.html').write_text(template.replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('</','<\\/')),encoding='utf-8')
    with (OUTPUT/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    print({k:len(v['cases']) for k,v in catalog.items()})


if __name__=='__main__':main()
