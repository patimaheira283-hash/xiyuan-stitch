"""Deterministic 4K-wide, three-input engineering acceptance (not a quality dataset)."""
from pathlib import Path
import argparse
import time
import cv2
import numpy as np
import psutil
from xiyuan_mvp.config import load_config
from xiyuan_mvp.image_io import read_image, write_image
from xiyuan_mvp.pipeline import StitchPipeline
from xiyuan_mvp.run_io import save_run, write_json

ROOT=Path(__file__).resolve().parents[1]


def create_scene():
    rng=np.random.default_rng(97)
    scene=cv2.resize(rng.integers(0,256,(225,512,3),np.uint8),(4096,1800))
    for i in range(120):
        x,y=map(int,rng.integers((40,40),(4050,1750)))
        cv2.circle(scene,(x,y),int(rng.integers(12,35)),(240,240,240),3)
        if i%6==0:cv2.putText(scene,str(i),(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(20,20,20),2)
    return scene


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--clear',action='store_true');parser.add_argument('--output',default='outputs/high-resolution-acceptance');args=parser.parse_args()
    folder=ROOT/args.output
    folder.mkdir(parents=True,exist_ok=True)
    scene=create_scene()
    inputs=[scene[:,:2200],scene[:,1100:3300],scene[:,2200:]]
    started=time.perf_counter()
    config=load_config();config['clear_fusion']['enabled']=args.clear
    result=StitchPipeline(config).run_many(inputs)
    save_run(folder/'run',result,config)
    write_image(folder/'export.png',result.final_image)
    assert read_image(folder/'export.png').shape==result.final_image.shape
    assert result.final_image.shape[1]>=4090
    assert result.metrics['sequence_count']==3
    assert len(result.source_images)==3
    write_json(folder/'acceptance.json',{'status':'passed','type':'procedural engineering stress case, no AI quality claim',
        'elapsed_seconds':time.perf_counter()-started,'input_count':3,'input_shapes':[list(i.shape) for i in inputs],
        'output_shape':list(result.final_image.shape),'process_peak_working_set_bytes':psutil.Process().memory_info().peak_wset,
        'metrics':result.metrics})


if __name__=='__main__':main()
