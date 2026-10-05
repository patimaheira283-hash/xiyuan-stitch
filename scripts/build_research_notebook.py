from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build(owner, output, experiment="v1"):
    source_paths = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT/"xiyuan_mvp").glob("*.py"))
    source_paths = [p for p in source_paths if p != "xiyuan_mvp/gui.py"]
    source_paths += ["xiyuan_mvp/default.yaml","scripts/research_gpu.py","data/research-cases.json"]
    if experiment in ("v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v15"):
        source_paths += ["scripts/refined_research_gpu.py","scripts/deformation_cases.py"]
    if experiment in ("v3", "v4", "v5"):
        source_paths += ["scripts/final_research_gpu.py", "scripts/high_resolution_acceptance.py"]
    if experiment in ('v4', 'v5'):
        source_paths += ['scripts/neural_research_gpu.py','scripts/prepare_neural_validation.py','data/neural-validation/sources.json']
    if experiment == 'v5':
        source_paths += ['scripts/hybrid_research_gpu.py']
    if experiment == "v6": source_paths += ["scripts/native_tiled_research_gpu.py"]
    if experiment == "v7": source_paths += ["scripts/hybrid_native_research_gpu.py"]
    if experiment == "v8": source_paths += ["scripts/hybrid_guard_research_gpu.py"]
    if experiment == "v9": source_paths += ["scripts/hybrid_gate_research_gpu.py"]
    if experiment == "v10": source_paths += ["scripts/hybrid_generation_tune_research_gpu.py"]
    if experiment == "v11": source_paths += ["scripts/hybrid_controlnet_scale_research_gpu.py"]
    if experiment == "v12": source_paths += ["scripts/hybrid_ablation_research_gpu.py"]
    if experiment == "v15": source_paths += ["scripts/raft_gate_tune_research_gpu.py"]
    entry={"v1":"research_gpu.py","v2":"refined_research_gpu.py","v3":"final_research_gpu.py","v4":"neural_research_gpu.py","v5":"hybrid_research_gpu.py","v6":"native_tiled_research_gpu.py","v7":"hybrid_native_research_gpu.py","v8":"hybrid_guard_research_gpu.py","v9":"hybrid_gate_research_gpu.py","v10":"hybrid_generation_tune_research_gpu.py","v11":"hybrid_controlnet_scale_research_gpu.py","v12":"hybrid_ablation_research_gpu.py","v15":"raft_gate_tune_research_gpu.py"}[experiment]
    sources = {p:(ROOT/p).read_text(encoding="utf-8") for p in source_paths}
    hashes = {p:hashlib.sha256(s.encode()).hexdigest() for p,s in sources.items()}
    def cell(kind, text, index):
        out={"cell_type":kind,"metadata":{},"id":f"research-{index}","source":text.splitlines(keepends=True)}
        if kind=="code":
            out.update(execution_count=None, outputs=[])
        return out
    bootstrap = (
        "import os,sys\nfrom pathlib import Path\n"
        "ROOT=Path('/kaggle/working/xiyuan_source');ROOT.mkdir(exist_ok=True)\n"
        f"SOURCES={sources!r}\n"
        "for relative,text in SOURCES.items():\n"
        " p=ROOT/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text,encoding='utf-8')\n"
        "sys.path.insert(0,str(ROOT))\n"
        "os.environ['TORCH_HOME']='/kaggle/temp/xiyuan-torch-cache'\n"
    )
    cells=[
        cell("markdown","# 曦源申请书算法对照实验\n\n私有免费 GPU 任务。LoFTR、普通 Inpainting、ControlNet Canny/Tile 与 LCM。\n"
             "开发集实验，保留失败记录，不预设优于传统算法。真实照片由作者公开链接下载且校验 SHA256，不包含凭据或申请书个人信息。",0),
        cell("code","import torch\nassert torch.cuda.is_available(), 'Free CUDA GPU is required'\n"
             "assert (torch.ones(4,device='cuda')*2).sum().item()==8\nprint(torch.cuda.get_device_name(0))\n",1),
        cell("code","import subprocess,sys\nsubprocess.run([sys.executable,'-m','pip','install','--quiet',"
             "'diffusers==0.35.1','transformers==4.56.2','accelerate==1.10.1','peft==0.17.1',"
             "'kornia==0.8.1','lpips==0.1.4','scikit-image==0.25.2','PyYAML>=6','Pillow>=10'],check=True,timeout=600)\n",2),
        cell("code",bootstrap,3),
        cell("code",f"import runpy\nrunpy.run_path(str(ROOT/'scripts/{entry}'),run_name='__main__')\n",4),
    ]
    notebook={"nbformat":4,"nbformat_minor":5,"cells":cells,
              "metadata":{"kernelspec":{"name":"python3","display_name":"Python 3","language":"python"}}}
    output.mkdir(parents=True,exist_ok=True)
    (output/"research.ipynb").write_text(json.dumps(notebook,ensure_ascii=False,indent=2),encoding="utf-8")
    (output/"source-manifest.json").write_text(json.dumps(hashes,indent=2),encoding="utf-8")
    metadata={"id":f"{owner}/xiyuan-stitch-research","title":"Xiyuan Stitch Research",
              "code_file":"research.ipynb","language":"python","kernel_type":"notebook",
              "is_private":True,"enable_gpu":True,"enable_tpu":False,"enable_internet":True,
              "machine_shape":"NvidiaTeslaT4","dataset_sources":[],"competition_sources":[],
              "kernel_sources":[],"model_sources":[]}
    (output/"kernel-metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print(f"Built {len(sources)} source files; no local credentials or personal documents.")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--owner",required=True)
    parser.add_argument("--output",type=Path,default=ROOT/"kaggle/research")
    parser.add_argument("--experiment",choices=["v1","v2","v3","v4","v5","v6","v7","v8","v9","v10","v11","v12","v15"],default="v1")
    args=parser.parse_args()
    build(args.owner,args.output,args.experiment)
