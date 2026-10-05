"""Portable three-route comparison, served by the existing local report server."""
from pathlib import Path
import json

from .hybrid_experiment import BASE, read
from .core import write_json


HTML = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>几何对齐 + AI 接缝修补 · 三方对照</title><style>
*{box-sizing:border-box}body{margin:0;background:#eef2f3;color:#172d37;font:14px/1.5 system-ui,"Microsoft YaHei",sans-serif}header{padding:12px 20px;background:white;border-bottom:1px solid #ccd6da}h1{font-size:22px;margin:0}p{margin:5px 0}.muted{font-size:13px;color:#516872}nav{display:flex;gap:8px;align-items:center;flex-wrap:wrap;padding:9px 20px;background:#dee9ed}button,select{font:inherit;border:1px solid #acbdc5;border-radius:5px;background:white;padding:6px 10px;color:inherit;cursor:pointer}select{max-width:55vw}main{padding:12px 20px}h2{margin:0 0 8px;font-size:18px}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.card{background:white;border:1px solid #cbd5da;border-radius:6px;overflow:hidden}.label{padding:7px 10px;background:#fff;border-bottom:1px solid #d2dce0;font-weight:650}.frame{height:350px;display:flex;align-items:center;justify-content:center;background:repeating-conic-gradient(#e2e7e9 0% 25%,#f1f4f5 0% 50%) 50%/16px 16px}.frame img{max-width:100%;max-height:100%;cursor:zoom-in;object-fit:contain}.note{padding:9px;font-size:13px;min-height:60px}.two{grid-template-columns:1fr 1fr;margin-top:10px}.two .frame{height:220px}details{margin-top:10px}a{color:#176a91}#audit{margin-top:10px;background:#e1eee7;padding:8px 12px;border-radius:5px}dialog{width:96vw;height:94vh;border:0;padding:0;border-radius:7px}dialog::backdrop{background:#172d37cf}.bar{padding:9px 12px;display:flex;gap:10px;align-items:center;background:#e1e9ed}#large{overflow:auto;height:calc(100% - 55px);background:#d9e0e4;text-align:center}#large img{max-width:100%;height:auto}#large.native img{max-width:none}body.inspect header{display:none}body.inspect .frame{height:420px}body.inspect .note{min-height:0}body.inspect h2{font-size:16px}body.inspect main,body.inspect nav{padding:6px 12px}@media(max-width:760px){.grid{grid-template-columns:1fr}.frame{height:300px}}
</style><header><h1>让 AI 只修接缝，会更好吗？</h1><p>传统拼接、整图直接生成、几何约束后的局部修补 · 固定 6 对样例</p><p class="muted">第三条路线在模型返回后用程序锁住保护区；“保护区不变”只证明没有额外改动，不代表原几何对齐已经正确。</p></header>
<nav><button id="prev">上一对</button><select id="cases" aria-label="选择测试对"></select><button id="next">下一对</button><button id="refresh">刷新结果</button><select id="third" aria-label="第三栏内容"><option value="safe">最终结果（含失败回退）</option><option value="hybrid">候选合成（验收前）</option><option value="raw">模型原始返回</option><option value="mask-guide">接缝范围（红色）</option><option value="aligned-candidate">对齐后的模型候选</option></select><a href="report.md">实验报告</a><a href="../index.html">第一轮完整对照</a></nav>
<main><h2 id="title"></h2><div class="grid"><section class="card"><div class="label">① 传统拼接基底</div><div class="frame" id="traditional"></div><div class="note" id="traditionalNote"></div></section><section class="card"><div class="label">② 直接给模型两张图</div><div class="frame" id="direct"></div><div class="note" id="directNote"></div></section><section class="card"><div class="label" id="thirdTitle">③ 几何对齐 + AI 修接缝 + 锁像素</div><div class="frame" id="result"></div><div class="note" id="resultNote"></div></section></div><p id="audit"></p><p id="review"></p><details><summary>查看两张原始输入</summary><div class="grid two"><div class="card"><div class="label">输入 A</div><div class="frame" id="left"></div></div><div class="card"><div class="label">输入 B</div><div class="frame" id="right"></div></div></div></details><p class="muted">点击图片可放大。真实照片没有完整参考图；视觉结论为 Codex 初评。生成失败或几何验收未通过会明确标示。</p></main>
<dialog id="zoom"><div class="bar"><strong id="zoomTitle"></strong><button id="native">原始像素 / 适应宽度</button><button id="close">关闭</button></div><div id="large"></div></dialog>
<script>let data=__DATA__;const $=id=>document.getElementById(id);let idx=Math.max(0,data.cases.findIndex(c=>c.number==Number(location.hash.slice(1)||1)));if(new URLSearchParams(location.search).has('inspect'))document.body.classList.add('inspect');
function zoom(src,label){$('large').replaceChildren();let im=new Image;im.src=src;im.alt=label;$('large').append(im);$('large').className='';$('zoomTitle').textContent=label;$('zoom').showModal()}
function photo(id,file,label){$(id).replaceChildren();if(!file){$(id).textContent='尚无可用结果';return}let im=new Image;im.src=file;im.alt=label;im.onclick=()=>zoom(file,label);$(id).append(im)}
function init(){$('cases').replaceChildren();data.cases.forEach((c,i)=>{let o=document.createElement('option');o.value=i;o.textContent=String(c.number).padStart(2,'0')+' · '+c.title;$('cases').append(o)});render()}
function render(){idx=Math.max(0,Math.min(data.cases.length-1,idx));let c=data.cases[idx];$('cases').value=idx;history.replaceState(null,'','#'+c.number);$('title').textContent=String(c.number).padStart(2,'0')+' · '+c.title;for(let key of ['traditional','left','right'])photo(key,c.folder+'/'+key+'.png',key);photo('direct',c.folder+'/gpt.png','直接融合原图');let mode=$('third').value;photo('result',c.files.includes(mode+'.png')?c.folder+'/'+mode+'.png':null,mode);$('thirdTitle').textContent=mode==='safe'?'③ 局部修补 · 通过验收才采用':$('third').selectedOptions[0].textContent;
$('traditionalNote').textContent=c.originalReview.traditional.note;$('directNote').textContent=c.originalReview.gpt.note;$('resultNote').textContent=c.review||'待视觉复核';let g=c.generation,h=c.hybrid,a=c.acceptance;$('audit').textContent=`允许编辑：${(c.mask.editable_fraction*100).toFixed(1)}% 有效画面。`+(g.status==='succeeded'?`生图 ${g.elapsed_seconds}s。`:`生成状态：${g.status||'等待'}。`)+(h.status==='applied'?`保护区改动 ${h.protected_pixel_changes} 像素；最终尺寸 ${h.size_actual.join('×')}。`:h.error||'尚未合成。');$('review').textContent=(a.status==='fallback'?'本例已拒绝采用模型候选，默认显示原传统结果。 ':'')+(c.conclusion||'');$('audit').style.background=a.status==='fallback'?'#ffe1cc':'#e1eee7';}
$('prev').onclick=()=>{idx--;render()};$('next').onclick=()=>{idx++;render()};$('cases').onchange=e=>{idx=Number(e.target.value);render()};$('third').onchange=render;$('refresh').onclick=async()=>{try{data=await(await fetch('data.json',{cache:'no-store'})).json();init()}catch(e){alert('请重新导出报告后再打开')}};$('close').onclick=()=>$('zoom').close();$('native').onclick=()=>$('large').classList.toggle('native');init();
</script></html>'''


QUALITY_PANEL = r'''<section id="qualityPanel" hidden>
<h3>这次修补，有没有变差？</h3>
<p class="muted">这里检查的是验收前的局部候选。最终回退图不会掩盖候选的问题。</p>
<p id="qualityStatus"></p><ul id="qualityAlerts"></ul><p id="qualityMetrics" class="muted"></p>
<p id="referenceStatus"></p>
<table id="referenceTable"><thead><tr><th>与参考图比较</th><th>传统结果</th><th>局部候选</th></tr></thead><tbody></tbody></table>
<p class="muted">相似度越接近 1 越相似；像素误差越小越好。这些指标不能单独证明文字、物体或场景真实。</p>
<details><summary>放大查看变化最多的局部</summary><p class="muted">同位置、同倍率裁剪。像素变化最多不等于错误最多；第三栏菜单可查看变化位置的红色标记。</p>
<div class="grid" id="detailGrid"><section class="card"><div class="label">传统结果 · 局部</div><div class="frame" id="detailBase"></div></section><section class="card"><div class="label">验收前候选 · 局部</div><div class="frame" id="detailCandidate"></div></section><section class="card" id="detailReferenceCard"><div class="label">参考图 · 局部</div><div class="frame" id="detailReference"></div></section></div></details>
<p class="muted">规则仍在探索中，可能漏检或误报。<a href="quality-report.md">查看局部复核说明</a></p></section>'''

QUALITY_SCRIPT = r'''
function renderQuality(c){
 const q=c.qualityAudit;$('qualityPanel').hidden=!q;if(!q)return;
 const d=q.diagnostics,r=q.reference,m=d.metrics;
 $('qualityStatus').textContent=d.label;
 $('qualityPanel').className=d.status==='risk'?'quality risk':'quality';
 $('qualityAlerts').replaceChildren();d.alerts.forEach(a=>{let li=document.createElement('li');li.textContent=a.message;$('qualityAlerts').append(li)});
 $('qualityMetrics').textContent=`新增近黑像素 ${(m.new_black_fraction*100).toFixed(1)}%；新增近白像素 ${(m.new_white_fraction*100).toFixed(1)}%；原有边缘未在附近找到的比例 ${m.missing_edge_fraction===null?'样本不足':(m.missing_edge_fraction*100).toFixed(1)+'%'}。`;
 $('referenceStatus').textContent=r.label;$('referenceTable').hidden=!r.available;
 let body=$('referenceTable').querySelector('tbody');body.replaceChildren();
 if(r.available){[['整图相似度','global_ssim'],['接缝相似度','seam_ssim'],['接缝像素误差（0–255）','seam_mae']].forEach(([label,key])=>{let tr=document.createElement('tr');[label,r.traditional[key].toFixed(3),r.candidate[key].toFixed(3)].forEach(value=>{let td=document.createElement('td');td.textContent=value;tr.append(td)});body.append(tr)})}
 photo('detailBase',c.folder+'/audit-base.png','传统结果的局部');photo('detailCandidate',c.folder+'/audit-candidate.png','验收前候选的局部');
 $('detailGrid').className=r.available?'grid':'grid two';$('detailReferenceCard').hidden=!r.available;
 photo('detailReference',r.available?c.folder+'/audit-reference.png':null,'参考图的局部');
}
'''

HTML = HTML.replace('</style>', r'''
#qualityPanel{margin:14px 0;padding:14px;background:white;border:1px solid #cbd5da;border-radius:6px}#qualityPanel.risk{border-color:#bb7439;background:#fff8f1}#qualityPanel h3{margin:0 0 5px;font-size:18px}#qualityStatus,#referenceStatus{font-weight:650}#qualityAlerts{padding-left:22px}#qualityAlerts:empty{display:none}#referenceTable{border-collapse:collapse;margin:10px 0;max-width:100%;font-variant-numeric:tabular-nums}#referenceTable th,#referenceTable td{text-align:left;padding:6px 12px;border:1px solid #cbd5da}#detailGrid{margin-top:10px}#detailGrid .frame{height:260px}#detailGrid .frame img{width:100%;height:100%;object-fit:contain}[hidden]{display:none!important}
</style>''')
HTML = HTML.replace('<p id="review"></p>', '<p id="review"></p>' + QUALITY_PANEL)
HTML = HTML.replace('<option value="aligned-candidate">', '<option value="audit-change-map">变化位置（红色仅表示像素变化）</option><option value="aligned-candidate">')
HTML = HTML.replace('function render(){', QUALITY_SCRIPT + '\nfunction render(){')
HTML = HTML.replace("$('audit').style.background=a.status==='fallback'?'#ffe1cc':'#e1eee7';}", "$('audit').style.background=a.status==='fallback'?'#ffe1cc':'#e1eee7';renderQuality(c);}")


def export(folder):
    spec = read(folder / "experiment.json")
    original = read(BASE / "reviews.json")
    reviews = read(folder / "reviews.json") if (folder / "reviews.json").exists() else {}
    rows = []
    for c in spec["cases"]:
        cell = folder / c["folder"]
        r = {**c, "originalReview": original[str(c["number"])], **reviews.get(str(c["number"]), {})}
        r["files"] = [p.name for p in cell.glob("*.png")]
        audit_file = cell / "quality-audit.json"
        if audit_file.exists():
            audit = read(audit_file)
            r["qualityAudit"] = {key: audit[key] for key in ["version", "diagnostics", "reference", "crop_xyxy"]}
        for key in ["generation", "hybrid", "acceptance"]:
            r[key] = read(cell / (key + ".json")) if (cell / (key + ".json")).exists() else {"status": "pending"}
            r[key] = {k: v for k, v in r[key].items() if k in ["status", "elapsed_seconds", "size_actual", "protected_pixel_changes", "editable_mae", "raw_aligned_protected_mae", "evaluation", "accepted", "reason"]}
        rows.append(r)
    data = {"id": spec["id"], "cases": rows}
    write_json(folder / "data.json", data)
    (folder / "index.html").write_text(HTML.replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")), encoding="utf-8")
    return data


if __name__ == "__main__":
    folder = Path(read(BASE / "hybrid-latest.json")["folder"])
    export(folder)
    print("Hybrid comparison exported", flush=True)
