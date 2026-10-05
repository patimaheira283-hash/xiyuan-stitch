"""Download only the pinned public evaluation photographs and verify SHA256."""
from pathlib import Path
import hashlib
import json
import urllib.request
from urllib.parse import quote

ROOT=Path(__file__).resolve().parents[1]


def main():
    cases=json.loads((ROOT/'data/structural-heldout.json').read_text(encoding='utf-8'))['cases']
    for case in cases:
        source=case['source']
        for item in case['files']:
            target=ROOT/'data/gpt-fusion-library'/item['file']
            if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()==item['sha256']:continue
            relative=item['file'].split('/',2)[2]
            url=f"https://raw.githubusercontent.com/{source['repository']}/{source['commit']}/{quote(relative)}"
            with urllib.request.urlopen(url,timeout=90) as response:content=response.read()
            if hashlib.sha256(content).hexdigest()!=item['sha256']:raise ValueError('Source checksum mismatch: '+item['file'])
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
            print(item['file'],flush=True)


if __name__=='__main__':main()
