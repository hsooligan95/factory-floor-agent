#!/usr/bin/env python3
"""Demo: the ContextVault interlock — runs remember each other.

Run 1 executes in this process with the optional ContextVault backend
enabled and writes its run summary into a fresh vault. Run 2 is a
FRESH PROCESS (run_agent.py as a subprocess, the way a scheduled or
handed-off run would start) — it opens the same vault, reads run 1's
summary back with its provenance before answering, and writes its own
summary as version 2. The vault's full history for the question's key
is printed at the end: two versions, each attributable to an agent, a
timestamp, and a run ID.

    python demo_context_vault.py

Requires the contextvault package, or the sibling context-vault
checkout (the backend finds its src/ automatically — see
src/context_vault_backend.py).
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from agent import run_agent  # noqa: E402
from context_vault_backend import _load_vault_class, question_key  # noqa: E402

QUESTION = "What is a state graph in LangGraph and how does state flow through it?"


def main() -> int:
    failures = []

    def check(cond, label):
        print(f"  {'✓' if cond else '✗ FAILED:'} {label}")
        if not cond:
            failures.append(label)

    with tempfile.TemporaryDirectory() as tmp:
        vault_path = str(Path(tmp) / "vault.db")
        print("=== Factory Floor × ContextVault demo ===")
        print(f"vault: {vault_path}")
        print(f"question: {QUESTION}\n")

        print("— run 1 (this process): no prior context, writes v1 —")
        first = run_agent(QUESTION, context_vault_path=vault_path)
        check(first["prior_run_summaries"] == [], "run 1 saw no prior summaries")
        check(first["context_vault"]["version"] == 1,
              f"run 1 wrote {first['context_vault']['key']} v1 "
              f"(source event {first['run_id']})")
        print(f"  answer grounded: {first['grounded']}")

        print("\n— run 2 (fresh process, run_agent.py): reads v1, writes v2 —")
        env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
        env["FACTORY_FLOOR_CONTEXT_VAULT"] = vault_path
        proc = subprocess.run([sys.executable, str(ROOT / "run_agent.py"), QUESTION],
                              capture_output=True, text=True, env=env, timeout=120)
        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr)
        check(proc.returncode == 0, "run 2 process exited cleanly")
        for line in proc.stdout.splitlines():
            if "Prior run context" in line or "written by" in line \
                    or "Run summary written" in line:
                print(f"  | {line}")
        check("Prior run context (ContextVault" in proc.stdout,
              "run 2 displayed the prior run's summary in its context")
        check(f"source event: {first['run_id']}" in proc.stdout,
              "the provenance run 2 saw is run 1's run ID")

        print("\n— the vault's history for this question —")
        vault = _load_vault_class()(vault_path)
        versions = vault.history(question_key(QUESTION))
        vault.close()
        for entry in versions:
            print(f"  {entry}")
        check([e.version for e in versions] == [1, 2],
              "two versions, nothing overwritten")
        check(all(e.agent and e.source_event for e in versions),
              "every version carries full provenance")

    print("\n=== DEMO OK — every step passed ===" if not failures
          else f"\n=== DEMO FAILED: {len(failures)} step(s) ===")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
