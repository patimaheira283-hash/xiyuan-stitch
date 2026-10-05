"""Build a private, self-contained Kaggle notebook for an exported desktop job."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build(job, owner, output, dataset=None):
    if job.stat().st_size > 30 * 1024**2:
        raise ValueError("自动提交任务暂限 30 MiB；更大任务请在 Kaggle 私有数据集中上传 ZIP。")
    sources = {p.relative_to(ROOT).as_posix(): p.read_text(encoding="utf-8")
               for p in (ROOT / "xiyuan_mvp").glob("*.py") if p.name != "gui.py"}
    sources["xiyuan_mvp/default.yaml"] = (ROOT / "xiyuan_mvp/default.yaml").read_text(encoding="utf-8")
    def cell(kind, source, i):
        value = {"cell_type": kind, "id": f"job-{i}", "metadata": {}, "source": source.splitlines(keepends=True)}
        if kind == "code": value.update(execution_count=None, outputs=[])
        return value
    bootstrap = ("from pathlib import Path\nimport os,sys\n"
                 "ROOT=Path('/kaggle/working/xiyuan_source');ROOT.mkdir(exist_ok=True)\n"
                 f"SOURCES={sources!r}\n"
                 "for name,text in SOURCES.items():\n p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text,encoding='utf-8')\n"
                 "sys.path.insert(0,str(ROOT))\n"
                 "jobs=list(Path('/kaggle/input').rglob('desktop-job.bin'))\n"
                 "assert len(jobs)==1, 'Attach your private desktop job dataset'\n"
                 "JOB=jobs[0]\n")
    cells = [cell("markdown", "# 曦源桌面接缝任务\n私有免费 GPU 推理。保留桌面配准与画笔区域，返回可直接打开的 run.json。", 0),
             cell("code", "import torch\nassert torch.cuda.is_available(), 'Enable free GPU first'\n"
                  "torch.cuda.set_per_process_memory_fraction(min(1,8*1024**3/torch.cuda.get_device_properties(0).total_memory))\n"
                  "import sys,subprocess\nsubprocess.run([sys.executable,'-m','pip','install','--quiet',"
                  "'diffusers==0.35.1','transformers==4.56.2','accelerate==1.10.1','peft==0.17.1','PyYAML>=6'],check=True)\n", 1),
             cell("code", bootstrap, 2),
             cell("code", "from xiyuan_mvp.gpu_job import run_job\nimport zipfile\n"
                  "output=Path('/kaggle/working/job-result')\n"
                  "try:\n result=run_job(JOB,output)\n print(result.metrics)\n"
                  "finally:\n with zipfile.ZipFile('/kaggle/working/job-result.zip','w',zipfile.ZIP_DEFLATED) as z:\n"
                  "  for p in output.rglob('*'):\n   if p.is_file():z.write(p,p.relative_to(output.parent))\n", 3)]
    output.mkdir(parents=True, exist_ok=True)
    (output / "job.ipynb").write_text(json.dumps({"nbformat":4,"nbformat_minor":5,"cells":cells,
        "metadata":{"kernelspec":{"name":"python3","display_name":"Python 3","language":"python"}}},ensure_ascii=False),encoding="utf-8")
    (output / "kernel-metadata.json").write_text(json.dumps({"id":f"{owner}/xiyuan-user-gpu-job", "title":"Xiyuan User GPU Job",
        "code_file":"job.ipynb","language":"python","kernel_type":"notebook","is_private":True,"enable_gpu":True,
        "enable_tpu":False,"enable_internet":True,"dataset_sources":[dataset] if dataset else [],"competition_sources":[],"kernel_sources":[]}),encoding="utf-8")
    (output / "source-manifest.json").write_text(json.dumps({k:hashlib.sha256(v.encode()).hexdigest() for k,v in sources.items()},indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("job", type=Path)
    p.add_argument("--owner", required=True)
    p.add_argument("--dataset")
    p.add_argument("--output", type=Path, default=ROOT / "kaggle/user-job")
    a = p.parse_args()
    build(a.job, a.owner, a.output, a.dataset)
