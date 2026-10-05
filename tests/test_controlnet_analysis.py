from scripts.analyze_controlnet_bundle import summarize_report


def test_successful_real_photos_count_without_inventing_quality():
    report = {"records": [
        {"id": "real", "profile": "canny", "status": "success", "quality": None,
         "metrics": {"inference_seconds": 2., "peak_vram_mb": None}},
        {"id": "broken", "profile": "canny", "status": "failed", "error": "OOM"},
    ]}
    row = summarize_report(report)["profiles"]["canny"]
    assert (row["attempted"], row["successes"], row["failures"]) == (2, 1, 1)
    assert row["reference_cases"] == 0 and row["without_reference"] == 1
    assert row["median_seam"]["lpips"] is None
    assert row["median_peak_vram_mb"] is None
    assert row["median_inference_seconds"] == 2.
    assert row["mean_delta_vs_structural"]["mae"] is None
    assert row["failures_detail"] == [{"id": "broken", "error": "OOM"}]
