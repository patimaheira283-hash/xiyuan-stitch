"""Frozen evaluation of protected RAFT on regression and disjoint source scenes."""
from pathlib import Path
from dataclasses import replace
import argparse
import hashlib
import json
import time
import cv2
import numpy as np
import torch
from scripts.validate_clear_fusion import make_cases
from xiyuan_mvp.structural_repair import repair,CLEAR
from xiyuan_mvp.clear_fusion import fuse
from xiyuan_mvp.neural_alignment import repair_alignment
from xiyuan_mvp.blending import feather_blend,poisson_blend
from xiyuan_mvp.registration import register_images
from xiyuan_mvp.config import load_config
from xiyuan_mvp.seam_mask import mask_from_binary
from xiyuan_mvp.benchmark import LPIPSEvaluator,masked_metrics
from xiyuan_mvp.image_io import read_image,write_image
from xiyuan_mvp.run_io import load_run,write_json,save_run
from xiyuan_mvp.types import RegistrationResult,StitchResult

ROOT=Path(__file__).resolve().parents[1]
LIBRARY=ROOT/'data/gpt-fusion-library'
PLAN=ROOT/'data/structural-heldout.json'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def prepare():
    if PLAN.exists():raise FileExistsError('The preselected source plan is already frozen')
    library=json.loads((LIBRARY/'manifest.json').read_text(encoding='utf-8'))
    provenance={r['file']:r for r in json.loads((LIBRARY/'provenance/images.json').read_text(encoding='utf-8'))['images']}
    previous=json.loads((ROOT/'data/research-cases.json').read_text())['cases']
    used_scenes={c['source_id'] for c in previous};used_hashes={f['sha256'] for c in previous for f in c['files']}
    chosen=[]
    for source in ['spw','lpc','rew','ges']:
        candidates=[c for c in library['cases'] if c['kind']=='real_capture' and c['source_id']==source
            and source+'/'+c['scene'] not in used_scenes and c.get('geometry_check',{}).get('status')=='supported']
        candidates.sort(key=lambda c:hashlib.sha256(('structural-050:'+c['id']).encode()).hexdigest())
        n=0
        for c in candidates:
            hashes=[provenance[c[k]]['sha256'] for k in ['left','right']]
            if any(h in used_hashes for h in hashes):continue
            files=[{'file':c[k],'sha256':h} for k,h in zip(['left','right'],hashes)]
            for item in files:assert sha(LIBRARY/item['file'])==item['sha256']
            chosen.append({'id':c['id'],'scene':source+'/'+c['scene'],'title':c['title'],'files':files,
                'kind':'real_capture','reference':None,'source':next(s for s in library['sources'] if s['id']==source)})
            used_hashes.update(hashes);used_scenes.add(source+'/'+c['scene']);n+=1
            if n==2:break
        assert n==2,source
    write_json(PLAN,{'selected_before_evaluation':True,'selection':'two scenes per repository; hash order; exclude prior v4 scenes and exact image duplicates',
        'limitations':'Scene IDs and hashes do not rule out all cross-repository near-duplicates. Sources were present in the local library; not previously used for this pipeline quality development.',
        'cases':chosen})
    print([c['id'] for c in chosen],flush=True)


def controlled_photo_cases(cases,*,joint=False):
    # Only A's original photograph supplies a withheld reference for this
    # controlled test; the real A/B pair is evaluated separately without scores.
    for source in cases:
        source_file=LIBRARY/source['files'][0]['file']
        assert sha(source_file)==source['files'][0]['sha256'],'Controlled source image changed'
        original=read_image(source_file);h,w=original.shape[:2]
        scale=min(1,640/max(h,w));ref=cv2.resize(original,(round(w*scale),round(h*scale)))
        h,w=ref.shape[:2];cut=int(w*.7);start=int(w*.3)
        for variant in (['joint_warp','joint_warp_light'] if joint else ['local_warp','light_gradient','blur_noise','noise']):
            for damaged in (['both'] if joint else ['a','b']):
                a=ref[:,:cut].copy();b=ref[:,start:].copy();im=a if damaged=='a' else b
                ih,iw=im.shape[:2];yy,xx=np.indices((ih,iw),dtype=np.float32)
                if joint:
                    for side,offset,direction in [(a,0,1),(b,start,-1)]:
                        yy,xx=np.indices(side.shape[:2],dtype=np.float32)
                        displacement=4*np.exp(-((xx+offset-(start+cut)/2)/(w*.12))**2)*np.sin(yy/(h/8)+.37)
                        side[:]=cv2.remap(side,xx,yy+displacement*direction,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
                    if variant=='joint_warp_light':b[:]=np.clip(b.astype(float)*np.linspace(.85,1.15,h)[:,None,None],0,255).astype(np.uint8)
                    changed=None
                elif variant=='local_warp':
                    center=(start+cut)/2-(0 if damaged=='a' else start)
                    displacement=8*np.exp(-((xx-center)/(w*.12))**2)*np.sin(yy/(h/8)+.37)
                    changed=cv2.remap(im,xx,yy+displacement,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
                elif variant=='light_gradient':changed=np.clip(im.astype(float)*np.linspace(.75,1.25,ih)[:,None,None],0,255).astype(np.uint8)
                elif variant=='noise':changed=np.clip(im.astype(float)+np.random.default_rng(8546).normal(0,14,im.shape),0,255).astype(np.uint8)
                else:changed=np.clip(cv2.GaussianBlur(im,(0,0),1.2).astype(float)+np.random.default_rng(7643).normal(0,2,im.shape),0,255).astype(np.uint8)
                if not joint:
                    if damaged=='a':a=changed
                    else:b=changed
                ca=np.zeros_like(ref);cb=ca.copy();ca[:,:cut]=a;cb[:,start:]=b
                ma=np.zeros((h,w),np.uint8);mb=ma.copy();ma[:,:cut]=255;mb[:,start:]=255
                reg=RegistrationResult(a,b,ca,cb,ma,mb,ma&mb,np.array([[1,0,start],[0,1,0],[0,0,1]],float),np.eye(3),np.zeros((8,8,3),np.uint8),{'feature_method':'known_crop_transform'})
                valid=np.full((h,w),255,np.uint8);valid=cv2.erode(valid,np.ones((7,7),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)
                yield {'id':source['id']+'_'+variant+'_'+damaged,'source':source['scene'],'variant':variant,'damaged_side':damaged},reg,ref,valid


def regression():
    for names,variants in [(['astronaut','chelsea','camera','immunohistochemistry'],['local_warp','light_gradient','blur_noise']),
                            (['text','coins','hubble_deep_field','moon'],['local_warp','light_gradient','blur_noise','noise']),
                            (['page','retina','cell','clock'],['local_warp','light_gradient','blur_noise','noise'])]:
        for info,reg,ref in make_cases(names,variants,640):
            valid=np.full(ref.shape[:2],255,np.uint8);valid[:3]=0;valid[-3:]=0;valid[:,:3]=0;valid[:,-3:]=0
            yield info,reg,ref,valid
    for case in json.loads((ROOT/'data/neural-validation/manifest.json').read_text())['cases']:
        saved,_=load_run(ROOT/'outputs/research/v4/extracted/research-results/baseline'/case['id'])
        reg=saved.registration;ref0=read_image(case['reference']);h,w=reg.canvas_a.shape[:2]
        ref=cv2.warpPerspective(ref0,reg.canvas_transform,(w,h))
        valid=cv2.warpPerspective(np.full(ref0.shape[:2],255,np.uint8),reg.canvas_transform,(w,h))
        valid=cv2.erode(valid,np.ones((7,7),np.uint8))
        yield {'id':case['id'],'source':case['source_id'],'variant':case['variant']},reg,ref,valid


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--suite',choices=['regression','heldout','joint','real'],default='regression');p.add_argument('--output');p.add_argument('--resume',action='store_true');args=p.parse_args()
    if args.prepare:return prepare()
    out=ROOT/(args.output or ('outputs/structural-050-'+args.suite));out.mkdir(parents=True,exist_ok=True)
    if (out/'results.json').exists() and not args.resume:raise FileExistsError('Use a new output directory, or resume the same frozen algorithm')
    torch.set_num_threads(3);cv2.setNumThreads(3)
    sources=json.loads(PLAN.read_text(encoding='utf-8'))['cases']
    signature=sha(ROOT/'xiyuan_mvp/structural_repair.py')
    protocol={'suite':args.suite,'algorithm_sha256':signature,'source_plan_sha256':sha(PLAN),
        'scope':('previously used regression sources' if args.suite=='regression' else 'additional joint-distortion stress evaluation on the same preselected sources; algorithm remains frozen' if args.suite=='joint' else 'preselected disjoint scene IDs and image hashes; frozen algorithm before evaluation'),
        'edit_mask':'entire overlap, feather radius 12','quality_region':'full valid reference area, 3px erosion',
        'real_input_max_size':1400,'controlled_reference_max_size':640,'source_plan':sources}
    if (out/'protocol.json').exists():
        prior=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
        assert prior['algorithm_sha256']==signature and prior['source_plan_sha256']==sha(PLAN)
    else:write_json(out/'protocol.json',protocol)
    (out/'executed-structural_repair.py').write_bytes((ROOT/'xiyuan_mvp/structural_repair.py').read_bytes())
    evaluator=None if args.suite=='real' else LPIPSEvaluator()
    cfg={'device':'cpu','max_size':512,'iterations':12,'consistency':1.5,'max_flow':24,'exposure':True,'preserve_detail':False}
    previous=json.loads((out/'results.json').read_text(encoding='utf-8')) if args.resume and (out/'results.json').exists() else {}
    rows=[r for r in previous.get('records',[]) if r['status']=='success']
    attempts=previous.get('failed_attempts',[])+[r for r in previous.get('records',[]) if r['status']!='success']
    completed={r['id'] for r in rows}
    if args.suite=='regression':items=regression()
    elif args.suite=='heldout':items=controlled_photo_cases(sources)
    elif args.suite=='joint':items=controlled_photo_cases(sources,joint=True)
    else:items=[(c,None,None,None) for c in sources]
    for info,reg,ref,valid in items:
        if info['id'] in completed:continue
        cid=info['id'];folder=out/cid;folder.mkdir(exist_ok=True);started=time.perf_counter()
        try:
            if args.suite=='real':
                inputs=[]
                for file in info['files']:
                    assert sha(LIBRARY/file['file'])==file['sha256']
                    im=read_image(LIBRARY/file['file']);h,w=im.shape[:2];s=min(1,1400/max(h,w));inputs.append(cv2.resize(im,(round(w*s),round(h*s))))
                config=load_config();config['registration'].update(method='loftr',loftr_device='cpu')
                reg=register_images(*inputs,config['registration'])
            base,bm=fuse(reg,CLEAR);mask=mask_from_binary(reg.overlap_mask,reg.overlap_mask,12)
            images={'clear04':base};metrics={'clear04':bm}
            images['protected'],metrics['protected']=repair(reg,base,mask,cfg)
            current_pipeline_seconds=time.perf_counter()-started
            if args.suite!='regression':
                images['raft04'],metrics['raft04']=repair_alignment(reg,base,mask,cfg,method='raft_large')
                images['Feather']=feather_blend(reg);images['Poisson']=poisson_blend(reg)
            quality={}
            for name,im in images.items():
                write_image(folder/(name+'.png'),im)
                if ref is not None:quality[name]={**masked_metrics(im,ref,valid),'lpips':evaluator.evaluate(im,ref,valid)}
            for name,im in [('a',reg.image_a),('b',reg.image_b)]:write_image(folder/(name+'.png'),im)
            if ref is not None:write_image(folder/'reference.png',ref)
            if args.suite=='real':
                config['clear_fusion']['enabled']=True
                config['inpainting']['engine']='neural_alignment';config['neural_alignment']['preserve_detail']=True
                result=StitchResult(base,images['protected'],reg,mask,{**metrics['protected'],'use_ai':True,'total_seconds':current_pipeline_seconds},images['Poisson'],reg.overlap_mask,[],base)
                save_run(folder/'run',result,config)
            assert np.array_equal(images['protected'][mask.soft_mask==0],base[mask.soft_mask==0])
            rows.append({**info,'status':'success','quality':quality or None,'metrics':metrics,'elapsed_seconds':time.perf_counter()-started})
            print(cid,{k:round(v['lpips'],5) for k,v in quality.items()} if quality else metrics['protected'],flush=True)
        except Exception as exc:
            import traceback
            rows.append({**info,'status':'failed','error':str(exc),'traceback':traceback.format_exc()})
            print(cid,'FAILED',str(exc),flush=True)
        write_json(out/'results.json',{'status':'running','records':rows,'failed_attempts':attempts})
    assert signature==sha(ROOT/'xiyuan_mvp/structural_repair.py'),'Algorithm changed during frozen evaluation'
    write_json(out/'results.json',{'status':'complete','records':rows,'failed_attempts':attempts})


if __name__=='__main__':main()
