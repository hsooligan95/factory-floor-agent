"""Emit-side AgentTrace event emission — optional, off by default.

When a trace ID is supplied via the environment — ``FACTORY_FLOOR_TRACE_ID``,
falling back to ``AGENTTRACE_TRACE_ID`` (the pattern agent-gate established
with ``AGENTGATE_TRACE_ID``) — ``run_agent`` appends a JSONL run record to
the path in ``FACTORY_FLOOR_TRACE_OUT`` (default:
``./factory-floor-trace.jsonl`` in the working directory).

The record is reshaped ONLY from the agent's final computed state: the
plan string the planner produced, the passages the retriever returned,
the groundedness checker's per-claim overlap scores and verdict, and the
answer/abstain outcome. Nothing is narrated. There is deliberately no
"options considered" field: this agent does not enumerate options, so
there is nothing honest to put in one. Its one computed decision point
is the groundedness check, and that event carries the checker's own
numbers (per-claim overlap, threshold, verdict) — the values that
decided the outcome, emitted as data.

This module is emit-side only: it reads the finished run state and
writes a file. It never feeds back into the graph. With no trace ID in
the environment, nothing is written and behaviour is exactly what it
was before this module existed.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

SOURCE = "factory-floor-agent"
DEFAULT_OUT = "factory-floor-trace.jsonl"


def trace_id_from_env(env: dict | None = None) -> str | None:
    """The externally supplied trace ID, if any (gate's env pattern)."""
    env = os.environ if env is None else env
    return env.get("FACTORY_FLOOR_TRACE_ID") or env.get("AGENTTRACE_TRACE_ID") or None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_run_record(result: dict, question: str, *, trace_id: str,
                     threshold: float) -> list[dict]:
    """Reshape a finished run's state into AgentTrace-shaped events."""
    def event(type_: str, payload: dict) -> dict:
        return {
            "trace_id": trace_id,
            "ts": _now(),
            "source": SOURCE,
            "agent": SOURCE,
            "type": type_,
            "payload": payload,
        }

    abstained = bool(result.get("abstained") or not result.get("grounded"))
    outcome = "abstained" if abstained else "answered"
    return [
        event("run_start", {"question": question}),
        event("plan", {
            "plan": result.get("plan"),
            "query": result.get("query"),
        }),
        event("retrieval", {
            "attempts": result.get("attempts", 0),
            "hits": [
                {"source": hit.chunk.source, "score": round(hit.score, 4)}
                for hit in result.get("hits", [])
            ],
        }),
        # The computed decision point: the groundedness checker's own
        # numbers, emitted as data at the end of the run that produced
        # them. (Per-claim sentences are the answer's own text; the
        # record keeps the scores and sources that decided the verdict.)
        event("groundedness", {
            "grounded": bool(result.get("grounded")),
            "threshold": threshold,
            "attempts": result.get("attempts", 0),
            "claims": [
                {
                    "overlap": claim["overlap"],
                    "grounded": claim["grounded"],
                    "source": claim["source"],
                }
                for claim in result.get("claims", [])
            ],
        }),
        event("outcome", {
            "outcome": outcome,
            "grounded": bool(result.get("grounded")),
            "abstained": abstained,
        }),
        event("run_end", {"outcome": outcome}),
    ]


def emit_run_record_if_configured(result: dict, question: str, *,
                                  threshold: float) -> Path | None:
    """Write the JSONL run record iff a trace ID was supplied.

    Returns the path written, or None when emission is not configured.
    """
    trace_id = trace_id_from_env()
    if not trace_id:
        return None
    out = Path(os.environ.get("FACTORY_FLOOR_TRACE_OUT", DEFAULT_OUT))
    events = build_run_record(result, question, trace_id=trace_id,
                              threshold=threshold)
    with out.open("a", encoding="utf-8") as fh:
        for event in events:
            fh.write(json.dumps(event) + "\n")
    return out
