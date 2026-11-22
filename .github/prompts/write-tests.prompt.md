---
mode: agent
description: Write tests for a named module in factory-floor-agent
---

Write tests for the module I name, following this repo's existing test
style, layout, and fixtures. Cover the behavior the module documents,
including these invariants where they apply:

- Answers compose ONLY from retrieved passages; groundedness is a computed token-overlap score, not a judgment call.
- An ungrounded answer retries once with wider retrieval, then abstains — never weaken the abstention path to make an answer appear.
- Retrieval returns top passages WITH their sources; citations are part of the output contract.

Then run the test suite:

    python -m unittest discover tests

Show the result. Do not claim done unless the suite is green; if a
test cannot pass, say why instead of weakening the assertion.
