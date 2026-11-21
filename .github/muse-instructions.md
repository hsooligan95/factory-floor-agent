# Copilot instructions — factory-floor-agent

You are working in the **Factory Floor Agent (LangGraph edition)**: a
runnable LangGraph RAG agent that answers questions about LangGraph,
LangChain, and Python from a bundled set of documentation notes, shows
the passages it used, and verifies its own answer before returning it.
It runs end to end offline — no API key required.

## Stack

- Python 3.10+; dependencies are `langgraph` and `langchain-core` only —
  retrieval and composition are standard library.
- `src/agent.py` (the graph), `src/retrieval.py` (hybrid search),
  `src/trace_emit.py` (optional AgentTrace emission). Entry:
  `run_agent.py`.

## Tests

```bash
python -m unittest discover tests   # 5 tests
python run_agent.py                 # offline demo question
```

## Architecture notes

- The graph is plan -> retrieve -> answer -> groundedness check. The
  answer composes ONLY from retrieved passages; groundedness is a
  computed token-overlap score, and an ungrounded answer retries once
  with a wider retrieval pass, then abstains. Never weaken the
  abstention path to make an answer appear.
- Demo mode is the default, not a fallback: with no `OPENAI_API_KEY`,
  a deterministic extractive composer builds the answer. Both paths
  pass the same groundedness check.
- Retrieval is hybrid BM25 + TF-IDF cosine blended in
  `src/retrieval.py`, returning top passages WITH their sources —
  citations are part of the output contract.
- Trace emission is inert unless `FACTORY_FLOOR_TRACE_ID` (fallback
  `AGENTTRACE_TRACE_ID`) is set. Emitted groundedness is computed
  data; never add narrative fields (e.g. "options considered") to
  emitted events.

## Shared rules

- Use Unix LF line endings in every file; never CRLF.
- Never commit secrets, tokens, or credentials — real or synthetic-looking.
- Tests must pass before you describe a change as complete; run the test
  command above and report honestly if you cannot.
- Prefer the smallest complete implementation: reuse what already exists in
  this repo before adding new code, files, or dependencies.
