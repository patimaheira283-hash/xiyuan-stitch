import ast
import json
import subprocess
import sys

from scripts.build_kaggle_notebook import ROOT, SOURCE_FILES, build_notebook


def test_notebook_is_self_contained_python_without_local_data():
    notebook, manifest = build_notebook()
    assert set(manifest) == set(SOURCE_FILES)
    for name in manifest:
        assert name.startswith(('xiyuan_mvp/', 'scripts/', 'configs/'))
        assert name.endswith(('.py', '.yaml'))
        assert (ROOT / name).is_file()
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            ast.parse(''.join(cell['source']))
            assert cell['outputs'] == []
    text = json.dumps(notebook)
    assert 'kaggle.json' not in text
    assert 'KAGGLE_KEY' not in text
    assert 'D:\\' not in text
    assert 'torch.cuda.is_available()' in text
    assert notebook['nbformat'] == 4


def test_bundled_sources_run_without_repository_files(tmp_path):
    for relative in SOURCE_FILES:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((ROOT / relative).read_text(encoding='utf-8'), encoding='utf-8')
    code = (
        'import sys; sys.path.insert(0, sys.argv[1]); '
        'from scripts.generate_demo_pair import main; main(); '
        'import xiyuan_mvp.cli; '
        'import xiyuan_mvp.neural_alignment, xiyuan_mvp.structural_repair, xiyuan_mvp.tiled_inpainting; '
        'from xiyuan_mvp.pipeline import StitchPipeline; '
        "r=StitchPipeline().run('data/demo_a.png','data/demo_b.png'); "
        "assert r.metrics['inliers'] >= 8; print('STANDALONE_OK')"
    )
    result = subprocess.run([sys.executable, '-I', '-c', code, str(tmp_path)],
                            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert 'STANDALONE_OK' in result.stdout
