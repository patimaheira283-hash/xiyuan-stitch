from pathlib import Path
import cv2
import numpy as np
from skimage import data
from xiyuan_mvp.image_io import write_image
from xiyuan_mvp.run_io import write_json


def create_cases(root: Path):
    cases=[]
    for name,loader in (("brick",data.brick),("grass",data.grass),("gravel",data.gravel),("coffee_wood",data.coffee)):
        raw=loader()
        reference=cv2.cvtColor(raw,cv2.COLOR_GRAY2BGR) if raw.ndim==2 else cv2.cvtColor(raw,cv2.COLOR_RGB2BGR)
        h,w=reference.shape[:2]
        a_end=int(w*0.665);b_start=int(w*0.335)
        a=reference[:,:a_end].copy()
        for index,(amplitude,gain,bias) in enumerate(((3,1.0,0),(8,1.0,0),(12,1.12,8),(8,0.82,-5)),1):
            b=reference[:,b_start:].copy()
            yy,xx=np.indices(b.shape[:2],dtype=np.float32)
            overlap=a_end-b_start
            bump=amplitude*np.exp(-((xx-overlap*0.50)/(overlap*0.30))**2)*np.sin(yy/(h/6.0))
            b=cv2.remap(b,xx,yy+bump,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
            b=np.clip(b.astype(float)*gain+bias,0,255).astype(np.uint8)
            case_id=f"{name}_deformation_{index}"
            folder=root/case_id
            write_image(folder/"a.png",a);write_image(folder/"b.png",b);write_image(folder/"reference.png",reference)
            cases.append({
                "id":case_id,"category":name,"source_id":name,"kind":"derived_photo_pair",
                "split":"development","image_a":str(folder/"a.png"),"image_b":str(folder/"b.png"),
                "reference":str(folder/"reference.png"),"reference_to_a":np.eye(3).tolist(),
                "deformation":{"max_vertical_pixels":amplitude,"gain":gain,"bias":bias},
                "note":"Controlled local nonrigid resampling of one public-domain/CC0 photograph; not an independent real camera pair.",
            })
    write_json(root/"manifest.json",{"cases":cases,"source_count":4,"description":"Development stress cases; no training or held-out quality claim."})
    return cases
