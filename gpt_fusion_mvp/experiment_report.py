"""Read-only local viewer and portable HTML export for the paired experiment."""
from __future__ import annotations

import argparse
from collections import Counter
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from statistics import mean, median
import platform
from urllib.parse import urlsplit

from .experiment import EXPERIMENTS, read
from .core import write_json


def collect(folder):
    spec = read(folder / "experiment.json")
    reviews = read(folder / "reviews.json") if (folder / "reviews.json").exists() else {}
    cases = []
    for case in spec["cases"]:
        row = {k: case[k] for k in ["number", "title", "case_id", "source_id", "kind", "folder", "reference"]}
        row["review"] = reviews.get(str(case["number"]), {})
        for method in ["traditional", "gpt"]:
            path = folder / case["folder"] / (method + ".json")
            raw = read(path) if path.exists() else {"status": "pending"}
            row[method] = {k: raw.get(k) for k in ["status", "elapsed_seconds", "size_actual", "quality_returned", "evaluation", "reused_prior_run", "size_matches_request"]}
            if raw.get("error"):
                error = raw["error"]
                row[method]["error"] = "HTTP 502：上游暂不可用" if "HTTP 502" in error else "连接中断，未返回图片" if any(x in error for x in ["SSLError", "Connection", "Timeout"]) else "请求失败，详见本地运行记录"
            row[method]["image"] = f"{case['folder']}/{method}.png" if raw.get("status") == "succeeded" else None
            row[method]["recovery_attempt"] = "recovery" in raw.get("id", "")
        cases.append(row)
    totals = {}
    for method in ["traditional", "gpt"]:
        ok = [c[method] for c in cases if c[method]["status"] == "succeeded"]
        times = [c["elapsed_seconds"] for c in ok]
        totals[method] = {"succeeded": len(ok), "total": len(cases), "mean_seconds": round(mean(times), 3) if times else None,
                          "median_seconds": round(median(times), 3) if times else None, "min_seconds": min(times) if times else None, "max_seconds": max(times) if times else None}
    return {"id": spec["id"], "cases": cases, "totals": totals}


HTML = r'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>23 对图像拼接对照实验</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f1f3f4;color:#15232d;font:15px/1.5 system-ui,"Microsoft YaHei",sans-serif}
header{padding:16px 26px;background:#fff;border-bottom:1px solid #d7dee2}h1{font-size:23px;margin:0}p{margin:5px 0}.muted{color:#536570;font-size:13px}
nav{display:flex;align-items:center;gap:9px;flex-wrap:wrap;padding:12px 26px;background:#e4edf0;position:sticky;top:0;z-index:2}
select,button,a.control{font:inherit;padding:7px 12px;border:1px solid #bccbd0;border-radius:5px;background:white;color:inherit}button,a{cursor:pointer}select{max-width:55vw}
main{padding:14px 26px}h2{font-size:19px;margin:0 0 5px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.card{min-width:0;background:#fff;border:1px solid #d7dee2;border-radius:6px;overflow:hidden}
.label{padding:7px 12px;border-bottom:1px solid #dce3e6;font-weight:600;font-size:14px;display:flex;justify-content:space-between;gap:8px}
.frame{background:repeating-conic-gradient(#e4e7e8 0% 25%,#f0f2f3 0% 50%) 50%/18px 18px;display:flex;align-items:center;justify-content:center;height:265px;overflow:auto}
.frame img{display:block;max-width:100%;max-height:100%;object-fit:contain;cursor:zoom-in}.outputs .frame{height:340px}.outputs{margin-top:12px}
.note{padding:9px 12px;min-height:44px;font-size:14px}.bad{color:#a12721}.ok{color:#12614e}.tag{white-space:nowrap;font-weight:400;color:#536570}
body.outputsonly .inputs{display:none}body.outputsonly .outputs .frame{height:65vh}body.inputsonly .outputs{display:none}body.inputsonly .inputs .frame{height:65vh}
dialog{width:96vw;height:94vh;padding:0;border:0;border-radius:8px}dialog::backdrop{background:#10202bcc}dialog .bar{padding:9px 15px;display:flex;gap:12px;align-items:center;background:#e4edf0}#large{overflow:auto;height:calc(100% - 57px);background:#d8dfe3;text-align:center}#large img{max-width:100%;height:auto}#large.native img{max-width:none}a{color:#176081}
@media(max-width:700px){.grid{grid-template-columns:1fr}main,header,nav{padding:10px}.frame{height:230px}.outputs .frame{height:260px}}
body.inspect header,body.inspect #kind,body.inspect .note,body.inspect main>p{display:none}body.inspect nav{padding:5px 14px}body.inspect main{padding:5px 14px}body.inspect .frame{height:230px}body.inspect .outputs .frame{height:295px}body.inspect .label{padding:3px 9px}body.inspect h2{font-size:16px}body.inspect.outputsonly .outputs .frame,body.inspect.inputsonly .inputs .frame{height:570px}
</style>
<header><h1>图像拼接对照实验 · 23 对</h1><p id="summary"></p><p class="muted">相同输入，最长边 1600 像素。传统：SIFT + RANSAC + 接缝 + 多频带融合；模型：两张照片直接送入 Codex 生图链路。</p></header>
<nav><button id="prev">上一对</button><select id="cases" aria-label="选择测试对"></select><button id="next">下一对</button><select id="layout" aria-label="查看布局"><option value="">全部对照</option><option value="outputsonly">放大看输出</option><option value="inputsonly">放大看输入</option></select><button id="refresh">刷新结果</button><a class="control" href="report.md">完整报告</a></nav>
<main><h2 id="title"></h2><p id="kind" class="muted"></p><div class="grid inputs"><section class="card"><div class="label">输入 A</div><div class="frame"><img id="left" alt="输入 A"></div></section><section class="card"><div class="label">输入 B</div><div class="frame"><img id="right" alt="输入 B"></div></section></div>
<div class="grid outputs"><section class="card"><div class="label"><span>传统拼接</span><span class="tag" id="traditionalMeta"></span></div><div class="frame" id="traditionalFrame"></div><div class="note" id="traditionalNote"></div></section><section class="card"><div class="label"><span>模型直接融合</span><span class="tag" id="gptMeta"></span></div><div class="frame" id="gptFrame"></div><div class="note" id="gptNote"></div></section></div>
<p class="muted" id="overall"></p><p class="muted">点击任意图片查看原图／原始像素。初评由 Codex 对照可见图像完成；真实照片没有完整参考图，此页不提供未经验证的准确率。</p><p id="reference"></p></main>
<dialog id="zoom"><div class="bar"><strong id="zoomTitle"></strong><button id="native">原始像素 / 适应宽度</button><button id="close">关闭</button></div><div id="large"></div></dialog>
<script>
let data=__DATA__, idx=Math.max(0,Number(location.hash.slice(1)||1)-1);const $=id=>document.getElementById(id);const inspecting=new URLSearchParams(location.search).has('inspect');if(inspecting)document.body.classList.add('inspect');
function zoom(src,title){$('large').replaceChildren();let im=new Image;im.src=src;im.alt=title;$('large').append(im);$('large').className='';$('zoomTitle').textContent=title;$('zoom').showModal()}
function init(){const s=$('cases');s.replaceChildren();data.cases.forEach((c,i)=>{let o=document.createElement('option');o.value=i;o.textContent=String(c.number).padStart(2,'0')+' · '+c.title;s.append(o)});let t=data.totals;$('summary').textContent=`传统出图 ${t.traditional.succeeded}/23 · 模型出图 ${t.gpt.succeeded}/23（包括复用样例）· 出图成功不等于内容保真`;render()}
function render(){idx=Math.max(0,Math.min(data.cases.length-1,idx));const c=data.cases[idx];$('cases').value=idx;history.replaceState(null,'','#'+c.number);$('title').textContent=String(c.number).padStart(2,'0')+' · '+c.title;$('kind').textContent=c.source_id+' · '+(c.reference?'受控样例，有完整参考图':'真实照片，无完整参考图');for(let k of ['left','right']){$(k).src=c.folder+'/'+k+'.png';$(k).onclick=()=>zoom($(k).src,k==='left'?'输入 A':'输入 B')}
for(let k of ['traditional','gpt']){let m=c[k],f=$(k+'Frame');f.replaceChildren();if(m.image){let im=new Image;im.src=m.image;im.alt=k==='gpt'?'模型直接融合原图':'传统拼接输出';im.onclick=()=>zoom(im.src,im.alt);f.append(im)}else{f.textContent=m.error||'等待结果';f.classList.add('bad')}
$(k+'Meta').textContent=m.status==='succeeded'?`${m.elapsed_seconds}s · ${(m.size_actual||[]).join('×')}${m.reused_prior_run?' · 复用':''}`:m.status;
let r=c.review[k];$(k+'Note').textContent=r?`${r.grade}：${r.note}`:'待视觉复核';}
$('overall').textContent=c.review.overall||'';$('reference').replaceChildren();if(c.reference){let a=document.createElement('a');a.href=c.folder+'/'+c.reference;a.textContent='查看完整参考图';a.onclick=e=>{e.preventDefault();zoom(a.href,'完整参考图')};$('reference').append(a)}}
$('prev').onclick=()=>{idx--;render()};$('next').onclick=()=>{idx++;render()};$('cases').onchange=e=>{idx=Number(e.target.value);render()};$('layout').onchange=e=>document.body.className=(inspecting?'inspect ':'')+e.target.value;$('close').onclick=()=>$('zoom').close();$('native').onclick=()=>$('large').classList.toggle('native');$('refresh').onclick=async()=>{try{data=await(await fetch('report-data.json',{cache:'no-store'})).json();init()}catch(e){alert('离线报告请重新打开最新 index.html。')}};
document.addEventListener('keydown',e=>{if($('zoom').open||['SELECT','INPUT'].includes(document.activeElement.tagName))return;if(e.key==='ArrowRight'){idx++;render()}if(e.key==='ArrowLeft'){idx--;render()}});init();
</script></html>'''


def export(folder):
    data = collect(folder)
    html = HTML.replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("<", "\\u003c"))
    if (folder / "hybrid-latest.json").exists():
        hybrid = Path(read(folder / "hybrid-latest.json")["folder"])
        if hybrid.parent.resolve() == folder.resolve() and (hybrid / "index.html").exists():
            html = html.replace('</nav>', f'<a class="control" href="{hybrid.name}/index.html">三方对照（6 对）</a></nav>')
    (folder / "index.html").write_text(html, encoding="utf-8")
    write_json(folder / "report-data.json", data)
    write_json(folder / "statistics.json", statistics(folder, data))
    return data


def statistics(folder, data):
    """Summarize only observed records; do not estimate dollar costs."""
    import cv2
    import numpy
    import PIL
    spec = read(folder / "experiment.json")
    records = [read(folder / c["folder"] / "gpt.json") for c in spec["cases"]]
    successful = [r for r in records if r["status"] == "succeeded"]
    new = [r for r in successful if not r.get("reused_prior_run")]
    initial_failures = [read(folder / c["folder"] / "gpt-initial-attempt.json") for c in spec["cases"] if (folder / c["folder"] / "gpt-initial-attempt.json").exists()]
    def times(values):
        return {"mean": round(mean(values), 3), "median": round(median(values), 3), "min": min(values), "max": max(values)} if values else None
    return {"experiment_id": data["id"], "totals": data["totals"], "new_successful_images": len(new),
            "reused_successful_images": len(successful)-len(new), "initial_failed_requests": len(initial_failures),
            "initial_error_types": dict(Counter("HTTP 502" if "HTTP 502" in r.get("error", "") else "SSL connection interrupted" for r in initial_failures)),
            "new_successful_request_seconds": times([r["elapsed_seconds"] for r in new]),
            "traditional_real_seconds": times([c["traditional"]["elapsed_seconds"] for c in data["cases"] if not c["reference"] and c["traditional"]["status"] == "succeeded"]),
            "visual_grades": {m: dict(Counter(c["review"].get(m, {}).get("grade", "待复核") for c in data["cases"])) for m in ["traditional", "gpt"]},
            "quality_returned": dict(Counter(q for r in successful for q in r.get("quality_returned", []))),
            "size_matched_count": sum(bool(r.get("size_matches_request")) for r in successful),
            "model_identity_verified_count": sum(bool(r.get("backend_identity_verified")) for r in successful),
            "reported_usage_new_successes": {k: sum(r.get("usage", {}).get(k, 0) for r in new if isinstance(r.get("usage"), dict)) for k in ["input_tokens", "output_tokens", "total_tokens"]},
            "usage_note": "返回 usage 不等于完整生图成本；失败请求是否扣费和总费用未知。",
            "environment": {"python": platform.python_version(), "platform": platform.platform(), "opencv": cv2.__version__, "numpy": numpy.__version__, "pillow": PIL.__version__}}


def serve(folder, port):
    class Handler(SimpleHTTPRequestHandler):
        extensions_map = {**SimpleHTTPRequestHandler.extensions_map, ".md": "text/plain; charset=utf-8"}

        def do_GET(self):
            if urlsplit(self.path).path == "/report-data.json":
                payload = json.dumps(collect(folder), ensure_ascii=False).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            else:
                super().do_GET()

        def list_directory(self, path):
            self.send_error(403, "Directory listing disabled")

        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, directory=str(folder)))
    print(f"Comparison viewer: http://127.0.0.1:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=18769)
    args = parser.parse_args()
    folder = args.folder or Path(read(EXPERIMENTS / "latest.json")["folder"])
    export(folder)
    if args.serve:
        serve(folder, args.port)
