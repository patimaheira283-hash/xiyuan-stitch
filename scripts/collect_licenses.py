from importlib.metadata import distribution, PackageNotFoundError
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]


def main():
    output=ROOT/'docs/delivery/third-party'
    output.mkdir(parents=True,exist_ok=True)
    records=[]
    for name in ('PySide6','PySide6_Essentials','PySide6_Addons','shiboken6','numpy','opencv-python','Pillow','PyYAML',
                 'PyInstaller','torch','torchvision','diffusers','transformers','kornia','kornia-rs','lpips','scikit-image',
                 'filelock','fsspec','networkx','sympy','mpmath','Jinja2','MarkupSafe','typing_extensions','safetensors'):
        try:d=distribution(name)
        except PackageNotFoundError:continue
        record={'name':name,'version':d.version,'license':d.metadata.get('License-Expression') or d.metadata.get('License'),
                'home_page':d.metadata.get('Home-page'),'files':[]}
        for item in d.files or []:
            if not any(k in item.name.lower() for k in ('license','copying','notice')):continue
            path=Path(d.locate_file(item))
            if not path.is_file() or path.suffix.lower() in ('.pyc','.py','.pyd','.dll'):continue
            target=output/name/str(item).replace('../','').replace('..\\','')
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,target)
            record['files'].append(target.relative_to(output).as_posix())
        records.append(record)
    (output/'DEPENDENCIES.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
