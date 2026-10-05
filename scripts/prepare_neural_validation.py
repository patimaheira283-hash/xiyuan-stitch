"""Fetch fixed CC0 texture assets, then derive explicit controlled overlap pairs."""
from pathlib import Path
import hashlib
import json
import cv2
import numpy as np
import requests
from xiyuan_mvp.image_io import read_image,write_image
from xiyuan_mvp.run_io import write_json

ROOT=Path(__file__).resolve().parents[1]
ASSETS=['fine_grained_wood','dark_wooden_planks','brick_wall_003','brick_wall_005']


def download_sources(root):
    records=[]
    for name in ASSETS:
        meta=requests.get('https://api.polyhaven.com/files/'+name,headers={'User-Agent':'XiyuanResearch/0.3'},timeout=60)
        meta.raise_for_status();metadata=meta.json();item=metadata['Diffuse']['1k']['jpg']
        target=root/'sources'/(name+'.jpg');target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            response=requests.get(item['url'],timeout=90);response.raise_for_status();target.write_bytes(response.content)
        raw=target.read_bytes()
        if hashlib.md5(raw).hexdigest()!=item['md5']:raise ValueError('Source MD5 mismatch: '+name)
        records.append({'id':name,'category':'wood' if 'wood' in name else 'brick','file':'sources/'+target.name,
            'url':item['url'],'sha256':hashlib.sha256(raw).hexdigest(),'md5':item['md5'],
            'asset_page':'https://polyhaven.com/a/'+name,'license':'CC0-1.0','license_url':'https://polyhaven.com/license',
            'kind':'public texture color map, not an independent camera overlap pair'})
    write_json(root/'sources.json',{'sources':records,'selection':'Four assets fixed before inspecting algorithm results on them.'})
    return records


def create_cases(root,source_manifest=None):
    sources=source_manifest or json.loads((root/'sources.json').read_text())['sources']
    cases=[]
    variants=[('lighting',0,1.18,8),('gradient',0,1,0),('local_5',5,1,0),('local_11',11,1,0),
              ('local_light',8,.82,-5),('perspective',0,1.12,5),('local_noise',8,1,0)]
    for source in sources:
        original=read_image(root/source['file']);h,w=original.shape[:2];scale=768/max(h,w)
        reference=cv2.resize(original,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
        h,w=reference.shape[:2];a_end=int(w*.68);b_start=int(w*.32)
        a=reference[:,:a_end]
        for i,(variant,amplitude,gain,bias) in enumerate(variants):
            b=reference[:,b_start:].copy();bh,bw=b.shape[:2]
            yy,xx=np.indices((bh,bw),dtype=np.float32)
            if amplitude:
                field=amplitude*np.exp(-((xx-(a_end-b_start)*.5)/((a_end-b_start)*.38))**2)*np.sin(yy/(h/7.3)+.37)
                b=cv2.remap(b,xx,yy+field,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
            b=np.clip(b.astype(float)*gain+bias,0,255).astype(np.uint8)
            if variant=='gradient':b=np.clip(b.astype(float)*np.linspace(.78,1.16,h)[:,None,None]+4,0,255).astype(np.uint8)
            if variant=='perspective':b=cv2.warpPerspective(b,np.array([[1,.014,2],[.008,1,1],[.000015,.00002,1]]),(bw,bh),borderMode=cv2.BORDER_REFLECT_101)
            if variant=='local_noise':b=np.clip(b.astype(float)+np.random.default_rng(20260910+i).normal(0,3,b.shape),0,255).astype(np.uint8)
            cid=source['id']+'_'+variant;folder=root/'cases'/cid
            write_image(folder/'a.png',a);write_image(folder/'b.png',b);write_image(folder/'reference.png',reference)
            cases.append({'id':cid,'source_id':source['id'],'category':source['category'],'variant':variant,
                'kind':'derived_texture_pair','split':'new source validation after six development cases',
                'image_a':str(folder/'a.png'),'image_b':str(folder/'b.png'),'reference':str(folder/'reference.png'),
                'reference_to_a':np.eye(3).tolist(),'source':source})
    write_json(root/'manifest.json',{'cases':cases,'independent_sources':len(sources),'case_count':len(cases),
        'warning':'Controlled crops of color maps. Not 28 independent real photographs.'})
    return cases


if __name__=='__main__':
    target=ROOT/'data/neural-validation';create_cases(target,download_sources(target));print('Prepared 28 cases from four new sources.')
