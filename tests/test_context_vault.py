"""ContextVault backend tests: the interlock with context-vault.

Disabled by default (no result keys added, no file created); enabled,
run 1 writes a versioned summary with full provenance, run 2 — a
fresh agent invocation, and in one test a fresh PROCESS — reads it
back with provenance intact. Provenance-free writes are impossible:
the vault rejects them, and this path never attempts one.

Run: python -m unittest discover tests
"""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent import run_agent  # noqa: E402
from context_vault_backend import _load_vault_class, question_key  # noqa: E402

QUESTION = "What is a state graph in LangGraph and how does state flow through it?"
ROOT = Path(__file__).resolve().parent.parent


class TestContextVaultBackend(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.vault_path = str(Path(self.tmp.name) / "vault.db")

    def tearDown(self):
        self.tmp.cleanup()

    def open_vault(self):
        return _load_vault_class()(self.vault_path)

    def test_disabled_by_default(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            result = run_agent(QUESTION)
        self.assertNotIn("prior_run_summaries", result)
        self.assertNotIn("context_vault", result)
        self.assertNotIn("run_id", result)
        self.assertFalse(Path(self.vault_path).exists())

    def test_run1_writes_summary_with_full_provenance(self):
        result = run_agent(QUESTION, context_vault_path=self.vault_path)
        self.assertEqual(result["prior_run_summaries"], [])
        self.assertEqual(result["context_vault"]["version"], 1)
        vault = self.open_vault()
        entry = vault.read(question_key(QUESTION), version=1)
        vault.close()
        self.assertEqual(entry.kind, "artifact")
        self.assertEqual(entry.agent, "factory-floor-agent")
        # The source event IS this run's ID — provenance, not narration.
        self.assertEqual(entry.source_event, result["run_id"])
        self.assertEqual(entry.value["question"], QUESTION)
        self.assertTrue(entry.value["grounded"])
        self.assertIn("LangGraph documentation", " ".join(entry.value["sources"]))

    def test_run2_reads_run1_summary_with_provenance(self):
        first = run_agent(QUESTION, context_vault_path=self.vault_path)
        second = run_agent(QUESTION, context_vault_path=self.vault_path)
        self.assertEqual(len(second["prior_run_summaries"]), 1)
        prior = second["prior_run_summaries"][0]
        self.assertEqual(prior["version"], 1)
        self.assertEqual(prior["agent"], "factory-floor-agent")
        self.assertEqual(prior["source_event"], first["run_id"])
        self.assertEqual(prior["summary"]["question"], QUESTION)
        self.assertTrue(prior["summary"]["grounded"])
        # Run 2's own write landed as the next version, based on v1.
        self.assertEqual(second["context_vault"]["version"], 2)
        vault = self.open_vault()
        history = vault.history(question_key(QUESTION))
        vault.close()
        self.assertEqual([e.version for e in history], [1, 2])
        self.assertEqual(history[1].base_version, 1)
        self.assertEqual(history[1].source_event, second["run_id"])

    def test_fresh_process_run_sees_prior_summary(self):
        first = run_agent(QUESTION, context_vault_path=self.vault_path)
        env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
        env["FACTORY_FLOOR_CONTEXT_VAULT"] = self.vault_path
        proc = subprocess.run(
            [sys.executable, str(ROOT / "run_agent.py"), QUESTION],
            capture_output=True, text=True, env=env, timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("Prior run context (ContextVault", proc.stdout)
        self.assertIn(f"source event: {first['run_id']}", proc.stdout)
        self.assertIn("v2", proc.stdout)  # the fresh run wrote version 2

    def test_write_without_provenance_is_rejected(self):
        vault = self.open_vault()
        with self.assertRaises(ValueError):
            vault.write("k", {"x": 1}, agent="", source_event="run-1")
        with self.assertRaises(ValueError):
            vault.write("k", {"x": 1}, agent="factory-floor-agent",
                        source_event="")
        vault.close()


if __name__ == "__main__":
    unittest.main()
