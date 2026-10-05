"""Create a reproducible GUI demonstration with known crop registration."""
from pathlib import Path
import time
from scripts.validate_clear_fusion import make_cases
from xiyuan_mvp.blending import feather_blend,poisson_blend
from xiyuan_mvp.clear_fusion import fuse
from xiyuan_mvp.structural_repair import CLEAR
from xiyuan_mvp.seam_mask import mask_from_binary
from xiyuan_mvp.types import StitchResult
from xiyuan_mvp.config import load_config
from xiyuan_mvp.run_io import save_run

ROOT=Path(__file__).resolve().parents[1]


def main():
    _,reg,_=make_cases(['astronaut'],['local_warp'],640)[0]
    started=time.perf_counter();base,metrics=fuse(reg,CLEAR)
    mask=mask_from_binary(reg.overlap_mask,reg.overlap_mask,12)
    config=load_config();config['clear_fusion']['enabled']=True;config['inpainting']['engine']='neural_alignment'
    all_metrics={**reg.metrics,**{'clear_'+k:v for k,v in metrics.items()},'clear_fusion_enabled':True,'use_ai':False,'total_seconds':time.perf_counter()-started}
    result=StitchResult(feather_blend(reg),base,reg,mask,all_metrics,poisson_blend(reg),reg.overlap_mask,[],base)
    save_run(ROOT/'data/structural-demo/motion-run',result,config)


if __name__=='__main__':main()
