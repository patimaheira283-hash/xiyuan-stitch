from xiyuan_mvp.config import load_config
from xiyuan_mvp.neural_alignment import repair_alignment
from xiyuan_mvp.structural_repair import CLEAR
from xiyuan_mvp.clear_fusion import fuse
from xiyuan_mvp.seam_mask import mask_from_binary
from tests.test_clear_fusion import pair


def test_protection_is_enabled_and_records_skipped_inference():
    cfg=load_config();reg,_=pair('b');base,_=fuse(reg,CLEAR)
    mask=mask_from_binary(reg.overlap_mask,reg.overlap_mask,12)
    result,metrics=repair_alignment(reg,base,mask,cfg['neural_alignment'],method='raft_large')
    assert cfg['neural_alignment']['preserve_detail']
    assert metrics['repair_reason']=='clear_source_already_selected'
    assert not metrics['model_executed']
