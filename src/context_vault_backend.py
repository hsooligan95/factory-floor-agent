"""Optional ContextVault backend — off by default.

When enabled, the agent's runs stop being one-shot: each run leaves a
versioned summary of itself in a ContextVault store (the sibling
context-vault project), and the next run on the same question reads
the prior summaries back — with their provenance — before it starts.

Enable it either way:

* constructor path: ``run_agent(question, context_vault_path=...)``
* environment:      ``FACTORY_FLOOR_CONTEXT_VAULT=<path to a vault db>``

One vault key per normalized question (``factory-floor/runs/<slug>``).
At run start the head version for the key is read as a PINNED version
(read by exact version number, so what the run saw is reproducible
from the vault's history). At run end the run writes its own summary
as the next version, declaring the version it read as its
``base_version`` — so if another writer moved the head in between,
the vault flags a conflict instead of silently merging, per its own
rules. Every write carries full provenance: agent
``factory-floor-agent``, and ``source_event`` = the run's ID (the
AgentTrace trace ID when one is configured, else a generated
``run-<hex>`` ID). The vault rejects writes without provenance, and
this backend never attempts one.

The vault is imported lazily, only when the backend is enabled: an
installed ``contextvault`` package if present, else the sibling
checkout's ``src`` directory next to this repo — the local bridge
between the two projects. With the backend disabled, nothing is
imported, read, or written, and run behaviour is exactly what it was
before this module existed.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from uuid import uuid4

from trace_emit import trace_id_from_env

AGENT_NAME = "factory-floor-agent"
KEY_PREFIX = "factory-floor/runs"


def vault_path_from_env(env: dict | None = None) -> str | None:
    env = os.environ if env is None else env
    return env.get("FACTORY_FLOOR_CONTEXT_VAULT") or None


def question_key(question: str) -> str:
    """One vault key per question: its normalized slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", question.lower()).strip("-")
    return f"{KEY_PREFIX}/{slug[:80].strip('-') or 'untitled'}"


def _load_vault_class():
    try:
        from contextvault import ContextVault
        return ContextVault
    except ImportError:
        sibling = (Path(__file__).resolve().parents[2]
                   / "context-vault" / "src")
        if sibling.is_dir() and str(sibling) not in sys.path:
            sys.path.insert(0, str(sibling))
        from contextvault import ContextVault  # raises if truly absent
        return ContextVault


class ContextVaultBackend:
    """Run-start reads and run-end summary writes against one vault."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._vault = _load_vault_class()(self.path)

    def close(self) -> None:
        self._vault.close()

    def prior_summaries(self, question: str) -> list[dict]:
        """The pinned head entry for this question's key, with its
        provenance — or [] when no prior run has written one."""
        key = question_key(question)
        head = self._vault.head_version(key)
        if head is None:
            return []
        entry = self._vault.read(key, version=head)  # pinned read
        return [{
            "key": entry.key,
            "version": entry.version,
            "summary": entry.value,
            "agent": entry.agent,
            "ts": entry.ts,
            "source_event": entry.source_event,
        }]

    def record_run(self, question: str, result: dict, *, run_id: str,
                   base_version: int | None) -> dict:
        """Write this run's summary as the next version of the key.

        Provenance is mandatory and always supplied: this agent's name
        and the run's own ID as the source event.
        """
        summary = {
            "question": question,
            "plan": result.get("plan"),
            "grounded": bool(result.get("grounded")),
            "abstained": bool(result.get("abstained")
                               or not result.get("grounded")),
            "attempts": result.get("attempts", 0),
            "answer": result.get("answer", ""),
            "sources": sorted({hit.chunk.source
                               for hit in result.get("hits", [])}),
        }
        entry = self._vault.write(
            question_key(question), summary,
            agent=AGENT_NAME, source_event=run_id, kind="artifact",
            base_version=base_version)
        return {"key": entry.key, "version": entry.version}


def new_run_id() -> str:
    """The run's ID: the configured AgentTrace trace ID if there is
    one, else a generated run ID. Either way it identifies this run
    as the source event of its vault write."""
    return trace_id_from_env() or f"run-{uuid4().hex[:12]}"
