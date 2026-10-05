import json
import zipfile

import cv2
import numpy as np
import pytest
import yaml

from xiyuan_mvp.config import load_config
from xiyuan_mvp.gpu_job import run_job, safe_extract
from xiyuan_mvp.pipeline import StitchPipeline
from xiyuan_mvp.run_io import save_run
from tests.test_pipeline_smoke import _synthetic_pair


def test_gpu_job_roundtrip_preserves_manual_mask_and_geometry(tmp_path):
    config=load_config()
    a,b=_synthetic_pair()
    prepared=StitchPipeline(config).run(a,b)
    save_run(tmp_path/'prepared',prepared,config)
    painted=prepared.mask.binary_mask.copy()
    painted[:painted.shape[0]//2]=0
    cv2.imwrite(str(tmp_path/'mask.png'),painted)
    job=tmp_path/'job.zip'
    with zipfile.ZipFile(job,'w') as archive:
        for p in (tmp_path/'prepared').iterdir():archive.write(p,'prepared/'+p.name)
        archive.write(tmp_path/'mask.png','mask.png')
        archive.writestr('config.yaml',yaml.safe_dump(config))
        archive.writestr('manifest.json',json.dumps({'schema_version':1,'cases':[{'prepared_run':'prepared/run.json','mask':'mask.png'}]}))
    result=run_job(job,tmp_path/'returned',use_ai=False)
    np.testing.assert_array_equal(result.mask.binary_mask,painted)
    np.testing.assert_array_equal(result.registration.homography_b_to_a,prepared.registration.homography_b_to_a)
    assert result.metrics['use_ai'] is False
    assert (tmp_path/'returned/run.json').is_file()


def test_gpu_job_rejects_zip_path_traversal(tmp_path):
    job=tmp_path/'job.zip'
    with zipfile.ZipFile(job,'w') as archive:archive.writestr('../escape.txt','bad')
    with pytest.raises(ValueError,match='非法路径'):safe_extract(job,tmp_path/'unpack')
    assert not (tmp_path/'escape.txt').exists()
