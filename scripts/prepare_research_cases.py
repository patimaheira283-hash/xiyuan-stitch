from __future__ import annotations
import json
from pathlib import Path
from urllib.parse import quote

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]


def main():
    library = ROOT / "data/gpt-fusion-library"
    manifest = json.loads((library / "manifest.json").read_text(encoding="utf-8"))
    recommendations = json.loads((library / "smoke-suite.json").read_text(encoding="utf-8"))["case_ids"]
    records = {x["file"]: x for x in json.loads((library / "provenance/images.json").read_text(encoding="utf-8"))["images"]}
    repositories = {
        "spw": "tlliao/Single-perspective-warps", "lpc": "dut-media-lab/Image-Stitching",
        "rew": "gain2217/Robust_Elastic_Warping", "ges": "flowerDuo/GES-GSP-Stitching",
    }
    candidates = [x for x in manifest["cases"] if x["kind"] == "real_capture" and x["source_id"] in repositories]
    candidates.sort(key=lambda x: (x["id"] not in recommendations, x["source_id"], x["scene"]))
    selected, seen = [], set()
    for case in candidates:
        scene = (case["source_id"], case["scene"])
        if scene in seen or case.get("review_required"):
            continue
        seen.add(scene)
        sid = case["source_id"]
        repo = repositories[sid]
        source = json.loads((library / "provenance" / (repo.replace("/", "__")+".json")).read_text(encoding="utf-8"))
        revision = source["tree"]["sha"]
        files = []
        for key in ("left", "right"):
            relative = case[key].split("/", 2)[2]
            files.append({
                "url": f"https://raw.githubusercontent.com/{repo}/{revision}/{quote(relative)}",
                "sha256": records[case[key]]["sha256"], "local_source": case[key],
            })
        selected.append({
            "id": case["id"].replace("-", "_"), "title": case["title"],
            "source_id": f"{sid}/{case['scene']}", "repository": repo, "revision": revision,
            "category": sid, "kind": "real_capture", "files": files,
            "reference": None, "license": "upstream research data; no standalone data license supplied",
            "source_note": "Download directly from the authors; retain attribution. Not licensed as CC0.",
        })
        if len(selected) == 30:
            break
    write_json(ROOT / "data/research-cases.json", {
        "schema_version": 1, "case_count": len(selected),
        "description": "30 real photographed pairs; no complete ground-truth panorama. Group by source scene; cross-repository similarity may remain.",
        "cases": selected,
    })
    print(f"Prepared {len(selected)} real pairs.")


if __name__ == "__main__":
    main()
