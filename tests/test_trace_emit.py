"""Emit-side AgentTrace record: off by default, on with a trace ID.

Run: python -m unittest discover tests
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent import run_agent  # noqa: E402

QUESTION = "What is a state graph in LangGraph and how does state flow through it?"


def _clean_env(**overrides):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("FACTORY_FLOOR_", "AGENTTRACE_"))}
    env.update(overrides)
    return env


class TestTraceEmit(unittest.TestCase):
    def test_no_trace_id_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "trace.jsonl"
            with patch.dict(os.environ, _clean_env(FACTORY_FLOOR_TRACE_OUT=str(out)),
                            clear=True):
                result = run_agent(QUESTION)
            self.assertTrue(result["grounded"])
            self.assertFalse(out.exists(), "no trace ID -> no run record")

    def test_record_written_with_trace_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "trace.jsonl"
            env = _clean_env(FACTORY_FLOOR_TRACE_ID="trace-test-123",
                             FACTORY_FLOOR_TRACE_OUT=str(out))
            with patch.dict(os.environ, env, clear=True):
                result = run_agent(QUESTION)
            self.assertTrue(result["grounded"])
            events = [json.loads(l) for l in out.read_text().splitlines()]
            self.assertEqual([e["type"] for e in events],
                             ["run_start", "plan", "retrieval",
                              "groundedness", "outcome", "run_end"])
            self.assertTrue(all(e["trace_id"] == "trace-test-123" for e in events))
            self.assertTrue(all(e["source"] == "factory-floor-agent" for e in events))
            retrieval = events[2]["payload"]
            self.assertTrue(retrieval["hits"], "expected retrieved passages recorded")
            self.assertTrue(all("source" in h and "score" in h
                                for h in retrieval["hits"]))
            groundedness = events[3]["payload"]
            self.assertTrue(groundedness["grounded"])
            self.assertEqual(groundedness["threshold"], 0.35)
            self.assertTrue(groundedness["claims"])
            for claim in groundedness["claims"]:
                self.assertIn("overlap", claim)
                self.assertIn("grounded", claim)
            # No fabricated narration: the record never claims the agent
            # enumerated options it did not enumerate.
            self.assertNotIn('"options"', json.dumps(events))
            self.assertEqual(events[4]["payload"]["outcome"], "answered")

    def test_generic_env_var_also_works_and_abstain_outcome(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "trace.jsonl"
            env = _clean_env(AGENTTRACE_TRACE_ID="trace-test-xyz",
                             FACTORY_FLOOR_TRACE_OUT=str(out))
            with patch.dict(os.environ, env, clear=True):
                run_agent("What is the best recipe for banana bread?")
            events = [json.loads(l) for l in out.read_text().splitlines()]
            self.assertTrue(all(e["trace_id"] == "trace-test-xyz" for e in events))
            self.assertEqual(events[4]["payload"]["outcome"], "abstained")
            self.assertFalse(events[3]["payload"]["grounded"])


if __name__ == "__main__":
    unittest.main()
