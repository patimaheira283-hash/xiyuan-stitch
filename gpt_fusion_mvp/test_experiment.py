"""Offline checks: never call the provider during validation."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from . import experiment as ex
from .core import Provider, sanitize, write_json


class ExperimentChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name) / "experiment"
        self.folder.mkdir()
        self.runs = Path(self.temp.name) / "runs"

    def spec(self, count=1):
        cases = []
        for n in range(1, count + 1):
            cell = self.folder / str(n)
            cell.mkdir()
            cases.append({"folder": str(n), "number": n, "case_id": str(n), "run_id": f"run{n}",
                          "inputs": [{"sha256": "a"}, {"sha256": "b"}], "reference": None, "prompt": "test"})
        write_json(self.folder / "experiment.json", {"cases": cases, "model_requested": "test", "mainline_model": "test", "quality": "high", "size": "1536x1024", "provider": {"id": "test"}})
        return cases

    def batch(self, retry=False):
        with patch.object(ex, "RUNS", self.runs), patch.object(ex, "list_runs", return_value=[]), contextlib.redirect_stdout(io.StringIO()):
            ex.gpt_batch(self.folder, workers=1, retry_failed=retry)

    def test_success_is_never_regenerated(self):
        self.spec()
        existing = {"status": "succeeded", "id": "first", "elapsed_seconds": 1}
        write_json(self.folder / "1/gpt.json", existing)
        with patch.object(ex, "run_fusion") as call:
            self.batch(retry=True)
            call.assert_not_called()
        self.assertEqual(ex.read(self.folder / "1/gpt.json"), existing)

    def test_recovery_preserves_failure_and_cannot_repeat(self):
        self.spec()
        initial = {"status": "failed", "id": "run1", "error": "HTTP 502"}
        write_json(self.folder / "1/gpt.json", initial)

        def fail(*args, **kwargs):
            result = {"status": "failed", "id": kwargs["run_id"], "error": "HTTP 502"}
            write_json(self.runs / kwargs["run_id"] / "run.json", result)
            return result

        with patch.object(ex, "run_fusion", side_effect=fail) as call:
            self.batch(retry=True)
            self.batch(retry=True)
            self.assertEqual(call.call_count, 1)
            self.assertEqual(call.call_args.kwargs["run_id"], "run1-recovery1")
        self.assertEqual(ex.read(self.folder / "1/gpt-initial-attempt.json"), initial)

    def test_stops_after_three_consecutive_failures(self):
        self.spec(6)
        with patch.object(ex, "run_fusion", return_value={"status": "failed", "error": "HTTP 502"}) as call:
            self.batch()
            self.assertEqual(call.call_count, 3)
        self.assertTrue(ex.read(self.folder / "progress.json")["stopped"])

    def test_transport_error_redacts_provider_and_key(self):
        p = Provider("id", "test", "https://private.invalid:123/v1", "secret-test-key")
        result = sanitize("host='private.invalid' https://private.invalid:123/v1 secret-test-key", p)
        self.assertNotIn("private.invalid", result)
        self.assertNotIn("secret-test-key", result)


if __name__ == "__main__":
    unittest.main()
