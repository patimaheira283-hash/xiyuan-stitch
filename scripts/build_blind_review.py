"""Create participant-only local review pages; keep assignment key separate."""
import json
from pathlib import Path
import random
import shutil
import cv2
from xiyuan_mvp.image_io import read_image, write_image
from xiyuan_mvp.run_io import load_run, write_json

ROOT=Path(__file__).resolve().parents[1]


def main():
    research=ROOT/'outputs/research/v3/extracted/research-results'
    cases=json.loads((research/'inputs-v3.json').read_text(encoding='utf-8'))['cases']
    output=ROOT/'deliverables/blind-review';images=output/'images';images.mkdir(parents=True,exist_ok=True)
    organizer=ROOT/'deliverables/盲评组织者材料';organizer.mkdir(parents=True,exist_ok=True)
    assignments=[];public=[];rng=random.Random(260910)
    for case in cases:
        if case.get('reference'):continue
        folder=research/'structural_only'/case['id']
        generated=research/'refined_canny'/case['id']/'result.png'
        if not generated.exists():continue
        baseline,_=load_run(folder)
        methods=[('Poisson',baseline.poisson_image),('refined_canny',read_image(generated))]
        rng.shuffle(methods);index=len(public)+1;cid=f'case_{index:03d}'
        bbox=baseline.mask.bbox
        for side,(method,image) in zip(('left','right'),methods):
            h,w=image.shape[:2];scale=min(1,1600/max(h,w))
            full=cv2.resize(image,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA) if scale<1 else image
            write_image(images/f'{cid}_{side}.png',full)
            x0,y0,x1,y1=bbox;x0=max(0,x0-64);y0=max(0,y0-64);x1=min(w,x1+64);y1=min(h,y1+64)
            write_image(images/f'{cid}_{side}_detail.png',image[y0:y1,x0:x1])
        public.append({'id':cid,'label':f'样例 {index:02d}'})
        assignments.append({'id':cid,'case_id':case['id'],'source_id':case['source_id'],
            'left':methods[0][0],'right':methods[1][0],'split':case['split']})
    write_json(organizer/'assignment-key.json',{'seed':260910,'comparisons':assignments})
    (organizer/'说明.md').write_text('# 盲评组织说明\n\n仅把 blind-review 文件夹发给评价者，不提供本目录。当前没有任何已完成的人类评分。\n\n收集下载的评分 JSON；按 assignment-key.json 解盲，按原始场景分组统计。每位评分者使用自选匿名编号，拒绝重复提交、空白项与无效项；同时报告平局和无法评价。左右位置固定随机化，避免在同一评价者处泄露方法。\n\n同一场景与开发集重叠情况保存在 key 中。评价者人数和偏好率必须由真实回收结果计算。\n',encoding='utf-8')
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>图像拼接盲评</title><style>
body{font:16px/1.6 system-ui,"Microsoft YaHei",sans-serif;margin:0;background:#f6f7f8;color:#18232c}main{max-width:1280px;margin:auto;padding:28px}h1{font-size:28px;margin:0}header{margin-bottom:24px}input,button,select,textarea{font:inherit;padding:8px;border:1px solid #bac5cd;border-radius:5px}button{cursor:pointer;background:#145866;color:white}article{background:white;padding:20px;margin:22px 0;border:1px solid #dde3e7;border-radius:6px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}.pair img{width:100%;height:360px;object-fit:contain;background:#17212a}.detail img{height:260px}fieldset{border:0;padding:8px 0}label{display:inline-block;margin:5px 14px 5px 0}textarea{display:block;width:96%;height:56px}.toolbar{display:flex;flex-wrap:wrap;gap:12px;align-items:center}.muted{color:#596b76}@media(max-width:720px){.pair{grid-template-columns:1fr}.pair img{height:260px}}
</style><main><header><h1>图像拼接盲评</h1><p>请比较左右图的接缝连续性、纹理保持和整体观感。方法名称已隐藏，没有标准答案；无法判断时请选择“无法评价”。</p><p class="muted">完整画布缩放至长边最多 1600 像素，接缝细节保留裁剪分辨率。可点击图片打开大图。</p><div class="toolbar"><label>匿名编号 <input id="participant" maxlength="40" placeholder="自选编号，不填姓名"></label><button id="export">下载评分 JSON</button><span id="progress" role="status"></span></div><p class="muted">页面保存在本地，不会向服务器发送评分。请下载后交给组织者。刷新或关闭前先下载。</p></header><div id="cases"></div></main><script>
const cases=__CASES__;
const criteria=[['seam','接缝连续性'],['texture','纹理保持'],['overall','整体偏好']];
const choices=[['left','左图更好'],['tie','相当'],['right','右图更好'],['unrateable','无法评价']];
document.querySelector('#cases').innerHTML=cases.map(c=>`<article data-case="${c.id}"><h2>${c.label}</h2><div class="pair">${['left','right'].map((s,i)=>`<div><strong>${i?'右图':'左图'}</strong><a href="images/${c.id}_${s}.png" target="_blank"><img src="images/${c.id}_${s}.png" alt="${c.label}${i?'右图':'左图'}完整画布" loading="lazy"></a></div>`).join('')}</div><details><summary>查看接缝细节</summary><div class="pair detail">${['left','right'].map(s=>`<a href="images/${c.id}_${s}_detail.png" target="_blank"><img src="images/${c.id}_${s}_detail.png" alt="${s}接缝细节" loading="lazy"></a>`).join('')}</div></details>${criteria.map(([key,label])=>`<fieldset><legend>${label}</legend>${choices.map(([v,t])=>`<label><input type="radio" name="${c.id}_${key}" value="${v}"> ${t}</label>`).join('')}</fieldset>`).join('')}<label for="${c.id}_note">可选说明</label><textarea id="${c.id}_note" maxlength="800"></textarea></article>`).join('');
function collect(){return cases.map(c=>({id:c.id,...Object.fromEntries(criteria.map(([k])=>[k,document.querySelector(`input[name="${c.id}_${k}"]:checked`)?.value??null])),note:document.getElementById(`${c.id}_note`).value}));}
function update(){const values=collect();document.getElementById('progress').textContent=`已完成 ${values.filter(v=>criteria.every(([k])=>v[k])).length} / ${cases.length} 组`;}
document.addEventListener('change',update);update();
document.getElementById('export').onclick=()=>{const participant=document.getElementById('participant').value.trim();if(!participant){alert('请填写匿名编号');return;}const result={schema_version:1,participant,created_utc:new Date().toISOString(),ratings:collect()};const blob=new Blob([JSON.stringify(result,null,2)],{type:'application/json'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='stitch-ratings-'+participant.replace(/[^a-zA-Z0-9_-]/g,'_')+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
</script></html>'''
    (output/'index.html').write_text(html.replace('__CASES__',json.dumps(public,ensure_ascii=False)),encoding='utf-8')
    print('Prepared',len(public),'unrated comparisons.')


if __name__=='__main__':main()
