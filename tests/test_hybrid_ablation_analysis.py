import pytest
from scripts.analyze_hybrid_ablation import summarize


def test_raft_gain_is_not_credited_to_diffusion_and_no_reference_rows_stay_unscored():
    quality = {stage: {'seam': {'mae': value*100, 'lpips': value, 'ssim': 1-value}}
               for stage, value in [('clear', .1), ('raft', .05), ('hybrid', .06), ('legacy_double_feather', .06)]}
    info = {'status': 'success', 'margin': 1.1, 'suite': 'regression',
            'metrics': {'diffusion_executed': True, 'model_executed': True}}
    data = {'records': [dict(info, id='controlled', quality=quality), dict(info, id='real', quality=None)]}
    group = summarize(data)['groups'][0]
    assert group['references'] == 1 and group['no_reference'] == 1
    assert group['hybrid_vs_clear']['metrics']['lpips']['improved'] == 1
    assert group['hybrid_vs_raft']['metrics']['lpips']['worse'] == 1
    assert group['hybrid_vs_raft']['metrics']['lpips']['mean_delta'] == pytest.approx(.01)
    assert group['hybrid_vs_raft']['paired_references'] == 1


def test_duplicate_success_cannot_inflate_paired_statistics():
    row = {'id': 'a', 'margin': 1.1, 'status': 'success'}
    with pytest.raises(ValueError, match='Duplicate'):
        summarize({'records': [row, row]})
