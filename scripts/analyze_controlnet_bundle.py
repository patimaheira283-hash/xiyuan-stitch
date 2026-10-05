"""Summarize a completed ControlNet GPU bundle without extracting its PNGs."""
from __future__ import annotations

import argparse
import json
import statistics
import zipfile
from collections import defaultdict
from pathlib import Path


def summarize_report(report: dict) -> dict:
    by_case = defaultdict(dict)
    for row in report.get("records", []):
        by_case[row["id"]][row["profile"]] = row
    profiles = sorted({row.get("profile") for row in report.get("records", []) if row.get("profile") != "structural_only"})
    output = {
        "experiment": report.get("experiment"),
        "gpu": report.get("gpu"),
        "record_count": len(report.get("records", [])),
        "profiles": {},
    }
    for profile in profiles:
        attempted = [cases[profile] for cases in by_case.values() if profile in cases]
        successful = [row for row in attempted if row.get("status") == "success"]
        rows = [
            row for row in successful if row.get("quality")
        ]
        values = {
            metric: [row["quality"][profile]["seam"][metric] for row in rows]
            for metric in ("mae", "lpips", "ssim")
        }
        profile_rows = [
            row for row in rows
            if row["id"] in by_case and by_case[row["id"]].get("structural_only", {}).get("quality")
        ]
        deltas = {
            metric: statistics.mean(
                row["quality"][profile]["seam"][metric]
                - by_case[row["id"]]["structural_only"]["quality"]["Structural"]["seam"][metric]
                for row in profile_rows
            ) if profile_rows else None
            for metric in values
        }
        output["profiles"][profile] = {
            "attempted": len(attempted),
            "successes": len(successful),
            "reference_cases": len(rows),
            "without_reference": len(successful) - len(rows),
            "failures": len(attempted) - len(successful),
            "median_seam": {metric: statistics.median(items) if items else None for metric, items in values.items()},
            "mean_seam": {metric: statistics.mean(items) if items else None for metric, items in values.items()},
            "mean_delta_vs_structural": deltas,
            "paired_reference_cases": len(profile_rows),
            "wins_vs_structural": {
                metric: sum((row["quality"][profile]["seam"][metric]
                    - by_case[row["id"]]["structural_only"]["quality"]["Structural"]["seam"][metric])
                    * (-1 if metric == "ssim" else 1) < -1e-8 for row in profile_rows)
                for metric in values
            },
            "failures_detail": [{"id": row["id"], "error": row.get("error")} for row in attempted if row.get("status") != "success"],
        }
        for metric in ("inference_seconds", "peak_vram_mb"):
            items = [row.get("metrics", {}).get(metric) for row in successful]
            items = [item for item in items if item is not None]
            output["profiles"][profile]["median_" + metric] = statistics.median(items) if items else None
    return output


def summarize(bundle: Path) -> dict:
    if bundle.suffix.lower() == ".json":
        report = json.loads(bundle.read_text(encoding="utf-8"))
    else:
        with zipfile.ZipFile(bundle) as archive:
            report = json.loads(archive.read("research-results/report-v2.json"))
    return summarize_report(report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(args.bundle)
    destination = args.output or args.bundle.with_name("v2-summary.json")
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(destination)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
