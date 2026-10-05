"""Package exactly Git HEAD, preserving its commit and per-file hashes."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def package(output, root=ROOT):
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing release: {output}")
    for args in (["diff", "--quiet"], ["diff", "--cached", "--quiet"]):
        subprocess.run(["git", *args], cwd=root, check=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    packed = subprocess.check_output(["git", "archive", "--format=zip", "HEAD"], cwd=root)
    with zipfile.ZipFile(io.BytesIO(packed)) as archive:
        files = {n: archive.read(n) for n in archive.namelist() if not n.endswith("/")}
    manifest = {n: hashlib.sha256(b).hexdigest() for n, b in sorted(files.items())}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(files.items()):
            archive.writestr(name, content)
        archive.writestr("SOURCE_MANIFEST.json", json.dumps(manifest, indent=2))
        archive.writestr("SOURCE_COMMIT.json", json.dumps({"commit": commit}, indent=2))
    result = {"file": output.name, "commit": commit, "files": len(files),
              "bytes": output.stat().st_size,
              "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    package(parser.parse_args().output)
