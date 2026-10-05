import numpy as np
import pytest
from xiyuan_mvp.config import load_config
from xiyuan_mvp.errors import StitchError
from xiyuan_mvp.pipeline import StitchPipeline
from tests.test_pipeline_smoke import _synthetic_pair


def test_structure_mode_routes_to_neural_repair_and_preserves_painted_mask(monkeypatch):
    a,b=_synthetic_pair();config=load_config();config['inpainting']['engine']='neural_alignment'
    pipe=StitchPipeline(config);prepared=pipe.run(a,b)
    painted=prepared.mask.binary_mask.copy();painted[:200]=0
    seen=[]
    def neural(registration,base,mask,settings,method):
        seen.append(mask.binary_mask.copy())
        return base.copy(),{'repair_engine':method,'test_double':True}
    def forbidden(*args,**kwargs):raise AssertionError('Structure mode must not start diffusion')
    monkeypatch.setattr('xiyuan_mvp.neural_alignment.repair_alignment',neural)
    monkeypatch.setattr('xiyuan_mvp.inpainting.DiffusersInpainter.generate',forbidden)
    result=pipe.run(a,b,prepared=prepared,use_ai=True,custom_mask=painted)
    np.testing.assert_array_equal(seen[0],painted)
    assert result.metrics['repair_engine']=='raft_large'
    assert result.metrics['use_ai'] is True
    prepared.metrics['sequence_count']=3
    with pytest.raises(StitchError,match='两图'):pipe.run(a,b,prepared=prepared,use_ai=True)


def test_hybrid_sends_neural_result_to_diffusion(monkeypatch):
    from xiyuan_mvp.inpainting import DiffusersInpainter
    a,b=_synthetic_pair();config=load_config();config['inpainting'].update(engine='hybrid', hybrid_diffusion_policy='always')
    config['blend']['color_match']=False
    config['blend']['ai_opacity']=.1
    pipe=StitchPipeline(config);prepared=pipe.run(a,b)
    neural_base=prepared.traditional_image.copy()
    neural_base[prepared.mask.binary_mask>0]=180
    seen=[]
    monkeypatch.setattr('xiyuan_mvp.neural_alignment.repair_alignment',lambda *a,**k:(neural_base.copy(),{'repair_engine':'raft_large'}))
    def generate(self,image,mask,**kwargs):
        seen.append(image.copy());self.last_metrics={'test_double':True};return image.copy()
    monkeypatch.setattr(DiffusersInpainter,'generate',generate)
    result=pipe.run(a,b,prepared=prepared,use_ai=True)
    assert seen and np.any(np.all(seen[0]==180,axis=2))
    np.testing.assert_array_equal(result.repair_base_image,neural_base)
    assert result.metrics['repair_engine']=='raft_large'
    assert 'neural_repair_seconds' in result.metrics
