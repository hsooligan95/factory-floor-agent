---
mode: agent
description: Review the current diff against factory-floor-agent's invariants
---

Review my current diff against this repo's invariants:

- Answers compose ONLY from retrieved passages; groundedness is a computed token-overlap score, not a judgment call.
- An ungrounded answer retries once with wider retrieval, then abstains — never weaken the abstention path to make an answer appear.
- Retrieval returns top passages WITH their sources; citations are part of the output contract.
- Trace emission is inert unless FACTORY_FLOOR_TRACE_ID / AGENTTRACE_TRACE_ID is set, and emitted events carry computed data only — no narrative fields.

Flag every violation with file and line. Also flag any change that
alters a documented behavior contract (output shapes, exit codes,
event schemas, recorded artifacts) without updating the docs and
tests that describe it. No style nitpicks — correctness and contract
violations only.
