from pathlib import Path
import json
import shutil
import hashlib
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'deliverables/structural-experiments-0.5.0'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    labels={'clear04':'0.4 清晰底图','protected':'0.5 保护结构修复','raft04':'旧版 RAFT 修复','Feather':'羽化融合','Poisson':'泊松融合'}
    groups=[('joint','两图同时错位 · 16 例','同一预选来源的附加压力实验；两图有相反的局部变形，算法在测试前已冻结。已知裁剪坐标，单独评价融合与修复。'),
            ('heldout','新来源受控案例 · 64 例','八个预选来源，各四种扰动、左右交替；与旧实验排除了相同场景编号与图片哈希，近似重复仍可能存在。算法在本组测试前冻结。已知裁剪坐标。'),
            ('real','真实照片 · 8 组','八组原始实拍图片经过 LoFTR 自动配准；无参考全景，不填写 SSIM / LPIPS。部分场景存在大视差。'),
            ('regression','旧数据回归 · 116 例','来自 16 张原始图片，已用于开发。88 例使用已知裁剪坐标；28 个纹理案例使用历史 LoFTR 画布。')]
    def copy(src,relative):
        target=OUT/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target);return relative
    suites={}
    for key,title,scope in groups:
        src=ROOT/('outputs/structural-050-'+key);report=json.loads((src/'results.json').read_text(encoding='utf-8'))
        assert report['status']=='complete' and all(r['status']=='success' for r in report['records'])
        protocol=json.loads((src/'protocol.json').read_text(encoding='utf-8'))
        assert protocol['algorithm_sha256']==hashlib.sha256((ROOT/'xiyuan_mvp/structural_repair.py').read_bytes()).hexdigest()
        raw=copy(src/'results.json',f'raw/{key}/results.json');proto=copy(src/'protocol.json',f'raw/{key}/protocol.json')
        copy(src/'executed-structural_repair.py',f'raw/{key}/structural_repair.py')
        cases=[]
        for row in report['records']:
            cid=row['id'];methods={};inputs={}
            for method,label in labels.items():
                p=src/cid/(method+'.png')
                if p.exists():methods[method]={'label':label,'image':copy(p,f'images/{key}/{cid}/{method}.png'),'quality':(row.get('quality') or {}).get(method)}
            for name in ['a','b','reference']:
                p=src/cid/(name+'.png')
                if p.exists():inputs[name]=copy(p,f'images/{key}/{cid}/{name}.png')
            cases.append({'id':cid,'methods':methods,'inputs':inputs,'metrics':row['metrics']['protected']})
        suites[key]={'title':title,'scope':scope,'raw':raw,'protocol':proto,'cases':cases}
    data={'version':'0.5.0','suites':suites}
    (OUT/'data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    template=(ROOT/'scripts/structural_viewer.html').read_text(encoding='utf-8')
    (OUT/'index.html').write_text(template.replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('</','<\\/')),encoding='utf-8')
    winners=[c for c in suites['joint']['cases'] if c['methods']['protected']['quality']['lpips']<c['methods']['clear04']['quality']['lpips']]
    chosen=winners[:2] or suites['joint']['cases'][:2]
    preview=Image.new('RGB',(1016,762),'#f3f5f4');draw=ImageDraw.Draw(preview)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',19)
    for col,label in enumerate(['0.4 清晰底图','旧版 RAFT','0.5 保护结构修复']):draw.text((16+col*334,12),label,font=font,fill='#18332e')
    for row,c in enumerate(chosen):
        label=c['id'].split('_joint')[0]+(' · 双侧变形＋光照变化' if '_light_' in c['id'] else ' · 双侧变形')
        draw.text((16,47+row*352),label,font=font,fill='#18332e')
        for col,key in enumerate(['clear04','raft04','protected']):
            im=Image.open(OUT/c['methods'][key]['image']).convert('RGB');w=min(320,im.width);h=min(320,im.height);x=(im.width-w)//2;y=(im.height-h)//2
            preview.paste(im.crop((x,y,x+w,y+h)),(16+col*334,77+row*352))
    preview.save(OUT/'comparison.png')
    print({k:len(v['cases']) for k,v in suites.items()})


if __name__=='__main__':main()
