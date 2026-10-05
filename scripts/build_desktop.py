"""Build the portable CPU desktop client. GPU inference lives in the source/GPU package."""
from pathlib import Path
import argparse
import os
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main(neural=False,clean=False,output=None):
    name='XiyuanAI' if neural else 'Xiyuan'
    destination=Path(output).resolve() if output else ROOT/('.work/portable-ai' if neural else '.work/portable')
    args=[sys.executable,"-m","PyInstaller","--noconfirm","--onedir","--windowed",
          "--name",name,"--paths",str(ROOT),
          "--add-data",str(ROOT/"xiyuan_mvp/default.yaml")+";xiyuan_mvp",
          "--distpath",str(destination),"--workpath",str(ROOT/".work/pyinstaller"),
          "--specpath",str(ROOT/".work")]
    excluded=["diffusers","transformers","peft","scipy","skimage","lpips","kaggle",
              'IPython','pytest','matplotlib','pandas','tensorboard','notebook','jupyter','tkinter']
    if not neural:excluded+=['torch','torchvision','kornia']
    for module in excluded:
        args += ["--exclude-module",module]
    if neural:
        import torch
        from xiyuan_mvp.neural_alignment import _raft
        _raft('cpu',True)
        checkpoint=Path(torch.hub.get_dir())/'checkpoints/raft_large_C_T_SKHT_V2-ff5fadd5.pth'
        args += ['--add-data',str(checkpoint)+';model_cache/hub/checkpoints','--collect-all','torchvision']
        alignment_checkpoint=Path(torch.hub.get_dir())/'checkpoints/loftr_outdoor.ckpt'
        if not alignment_checkpoint.is_file():
            from xiyuan_mvp.loftr import _model
            _model('outdoor','cpu','')
        args += ['--add-data',str(alignment_checkpoint)+';model_cache/hub/checkpoints',
                 '--collect-all','kornia','--copy-metadata','torch','--copy-metadata','torchvision','--copy-metadata','kornia']
    args.append(str(ROOT/"scripts/desktop_entry.py"))
    # Poppler's ICU on a document-runtime PATH has the same filename as Windows
    # ICU but different symbols. Do not let the freezer collect that unrelated DLL.
    build_env = os.environ.copy()
    build_env["PATH"] = os.pathsep.join(part for part in build_env.get("PATH", "").split(os.pathsep)
                                        if "poppler" not in part.lower())
    if clean:args.insert(3, "--clean")
    subprocess.run(args,cwd=ROOT,env=build_env,check=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--neural',action='store_true')
    parser.add_argument('--clean',action='store_true')
    parser.add_argument('--output',type=Path)
    options=parser.parse_args()
    main(options.neural,options.clean,options.output)
