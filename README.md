# Factory Floor Agent — LangGraph Edition

A runnable **LangGraph agent with retrieval (RAG)** that answers questions
about LangGraph, LangChain, and Python — using only a bundled set of public
documentation notes, showing the exact passages it used, and verifying its
own answer before returning it. It runs **end to end offline**: no API key
is required.

This is the LangGraph edition of the Factory Floor Agent line: the same
plan → retrieve → answer → verify discipline as the dependency-free CLI
edition, rebuilt as an explicit LangGraph state machine.

## Quick start (demo mode — no API key needed)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_agent.py
```

That runs the built-in demo question fully offline. The default mode *is*
the demo mode: with no `OPENAI_API_KEY` set, a deterministic extractive
composer builds the answer from the retrieved passages, so anyone can
clone this repo and watch the full graph — plan, retrieval, answer,
groundedness check — execute with zero credentials. Ask it your own
question by passing it as an argument:

```bash
python run_agent.py "What is a state graph in LangGraph and how does state flow through it?"
```

Requires Python 3.10+. Dependencies: `langgraph` and `langchain-core`
(the retrieval and composition layers are standard library only).

## What it does

1. **Plan / route** — classifies the question (concept, how-to, comparison,
   code-example) and forms the retrieval query.
2. **Retrieve** — chunks the document set and runs **hybrid search**:
   BM25 keyword scoring blended with TF-IDF vector-lite cosine similarity
   (standard library only), returning the top passages **with sources**.
3. **Answer** — composes an answer **only from the retrieved passages**
   (deterministic extractive composer offline; an optional
   OpenAI-compatible provider is used instead if `OPENAI_API_KEY` is set,
   and its output still has to pass the same check).
4. **Groundedness self-check** — verifies every answer sentence traces to
   a retrieved passage. If a claim is unsupported, the graph loops back to
   retrieve again; if it still cannot ground the answer, it **abstains**
   instead of guessing.

## Example run (real output, captured from this repo)

```text
$ python run_agent.py
Question: What is a state graph in LangGraph and how does state flow through it?
Plan: concept lookup

Retrieved passages:
  [1] source: LangGraph documentation - Graph API / StateGraph (hybrid score 0.93)
  [2] source: LangGraph documentation - Overview (hybrid score 0.93)
  [3] source: LangGraph documentation - Graph API / StateGraph (hybrid score 0.86)
  [4] source: LangGraph documentation - Overview (hybrid score 0.70)

Answer:
  LangGraph merges that update into the state before the next node runs, so
  state flows through the graph step by step and every later node sees the
  work of earlier nodes. Nodes are Python functions that do work, edges
  define how control flows between nodes, and a shared state object is
  passed along and updated as execution moves through the graph. ...

Groundedness check:
  [GROUNDED overlap=1.0] LangGraph merges that update into the state before ...
  [GROUNDED overlap=1.0] Nodes are Python functions that do work, edges ...
  [GROUNDED overlap=1.0] You define a graph by declaring a state schema, ...
  [GROUNDED overlap=1.0] The state schema is commonly a typed dictionary ...
  Result: grounded
```

The full unabridged run — every passage, the complete answer, and the
check — is recorded in [`tests/expected_run.md`](tests/expected_run.md),
along with the abstention case: ask about something outside the corpus
(for example, a banana bread recipe) and the agent retrieves nothing
relevant, fails its own check, and answers *"I could not ground an answer
to that question in the document set, so I am abstaining rather than
guessing."*

Tests: `python -m unittest discover tests` (10 tests: grounded answer +
abstention, the AgentTrace emitter's off-by-default / on-with-ID
behaviour, and the ContextVault backend's write/read-back cycle —
all pass).

## Architecture

```text
START -> plan -> retrieve -> answer -> groundedness_check
                    ^                        |
                    +--- retry (conditional edge, max 2 attempts)
                                             +-> END (grounded or abstain)
```

- **State** (`AgentState`, a `TypedDict`): `question`, `plan`, `query`,
  `hits`, `answer`, `claims`, `grounded`, `attempts`, `abstained`.
- **Nodes**: `plan`, `retrieve`, `answer`, `groundedness_check`.
- **Conditional edge**: after the check, the graph routes back to
  `retrieve` for one broader retry (wider top-k on the same query), or
  ends.
- **Retrieval as a tool**: the retriever sits behind a small, explicit
  interface (`HybridIndex.search(query, top_k) -> list[Hit]`), so the
  graph invokes it the way an agent invokes a tool — and the TF-IDF
  vector-lite index can be swapped for a real embedding/vector-store
  retriever without touching the graph.
- **Corpus**: 16 public documentation notes in `docs/` (LangGraph,
  LangChain, and Python docs), with attribution in
  `docs/ATTRIBUTION.md`. Nothing private is in the corpus or the index.

## Configuration

No configuration is required to run offline. The optional LLM provider
path is configured purely through environment variables (see
[`.env.example`](.env.example)):

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | If set, an OpenAI-compatible provider composes answers instead of the offline composer. Its output still must pass the groundedness check. |
| `OPENAI_MODEL` | Model name for the provider path (default `gpt-4o-mini`). |
| `OPENAI_BASE_URL` | Provider endpoint (default `https://api.openai.com/v1`). |

Keys are read from the environment only — never from code, never from a
file in this repo. On any provider failure the agent falls back to the
offline composer.

### AgentTrace run records (optional, off by default)

Supply a trace ID and each run appends a JSONL run record — run start,
plan, retrieval, the groundedness checker's per-claim scores and
verdict, and the answer/abstain outcome — reshaped only from the run's
computed state (no narration; the record contains no "options
considered" claim, because this agent enumerates none):

| Variable | Purpose |
|---|---|
| `FACTORY_FLOOR_TRACE_ID` | Trace ID to record under. `AGENTTRACE_TRACE_ID` is accepted as a fallback. Unset → nothing is written and behaviour is unchanged. |
| `FACTORY_FLOOR_TRACE_OUT` | Output path (default `./factory-floor-trace.jsonl`). |

The emitter (`src/trace_emit.py`) is emit-side only: it reads the
finished state and never feeds back into the graph.

### Interlock: ContextVault run summaries (optional, off by default)

With a [ContextVault](https://github.com/hsooligan95/context-vault)
store configured, runs stop being one-shot: at run start the agent
reads the prior run summary for the same question — a pinned version,
with its provenance (which agent wrote it, when, from which run) —
and at run end it writes its own summary as the next version,
declaring the version it read as its base so a moved head surfaces
as a vault conflict rather than a silent merge. Every write carries
full provenance: agent `factory-floor-agent`, source event = the
run's ID (the AgentTrace trace ID when configured, else a generated
`run-<hex>` ID). One vault key per normalized question:
`factory-floor/runs/<slug>`.

| Variable / argument | Purpose |
|---|---|
| `FACTORY_FLOOR_CONTEXT_VAULT` | Path to a ContextVault SQLite file. Unset → backend off, behaviour unchanged. |
| `run_agent(question, context_vault_path=...)` | The same, per call; wins over the env var. |

The backend (`src/context_vault_backend.py`) imports the vault
lazily, only when enabled: an installed `contextvault` package, or
the sibling context-vault checkout's `src/` — the local bridge
between the two projects. `python demo_context_vault.py` runs the
story end to end (run 2 is a fresh process reading run 1's summary);
captured output is in `demo-context-vault-output.txt`.

## What this demonstrates

- **LangGraph state machines** — a real `StateGraph` with typed state,
  four nodes, and a conditional retry edge, compiled and invoked via
  LangGraph (`src/agent.py`).
- **Agent orchestration** — a staged plan → retrieve → answer → verify
  pipeline where each stage's output is explicit, inspectable state, not
  hidden prompt glue.
- **Tool calling / retrieval as a tool** — hybrid retrieval exposed
  behind a tool-shaped interface and invoked by the graph as a distinct
  step, with scores and sources preserved into state.
- **Retrieval engineering (RAG)** — paragraph chunking with overlap,
  BM25 + TF-IDF hybrid scoring with per-query normalization, and a
  relevance gate that requires real token overlap before a passage counts
  as evidence (`src/retrieval.py`).
- **Multi-step execution with a loop** — the groundedness check can route
  the graph back for a second, wider retrieval pass before giving up.
- **Reliability of LLM output** — every answer sentence is verified
  against the retrieved passages; ungrounded answers trigger a retry and
  then an explicit abstention rather than a confident guess.
- **Observability** — every run prints its plan, the retrieved passages
  with sources and hybrid scores, and a per-sentence groundedness report,
  so a reviewer can audit exactly why an answer was returned.
- **LLM API integration, safely** — an optional OpenAI-compatible provider
  behind the same interface, with offline fallback and the same
  verification applied to its output.

## Repository layout

```text
run_agent.py          CLI entry point (one run command)
src/agent.py          LangGraph state machine (4 nodes + conditional edge)
src/retrieval.py      Chunking + hybrid BM25 / TF-IDF retrieval
src/trace_emit.py     Optional AgentTrace run-record emitter (off by default)
src/context_vault_backend.py  Optional ContextVault run-summary backend (off by default)
docs/                 16 public documentation notes + ATTRIBUTION.md
tests/test_agent.py   Smoke tests (grounded answer + abstention)
tests/test_trace_emit.py  Emitter contract tests (off by default, on with ID)
tests/test_context_vault.py  ContextVault backend tests (off by default, write/read-back, fresh process)
demo_context_vault.py The ContextVault interlock demo
tests/expected_run.md Recorded real run with sources visible
requirements.txt      langgraph, langchain-core
.env.example          Optional provider configuration (env vars only)
```

## Deployment notes (NOT deployed — notes only)

Nothing in this repository is deployed anywhere. If it were deployed,
the notes in `docs/16-deployment-notes.md` describe the intended shape:
run the compiled graph behind a small service, supply provider keys via
environment variables / a secret manager (never in code), package with a
container image built from the pinned `requirements.txt`, and log the
plan, retrieved sources, and check result per run for observability.
AWS is one possible host (e.g. a container service or Lambda behind an
API) — this is a design note, not a running system.

## License

MIT — see [LICENSE](LICENSE). The `docs/` corpus is derived from public
LangGraph, LangChain, and Python documentation; see
`docs/ATTRIBUTION.md` for sources and terms.
