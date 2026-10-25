## Code Review: quoted two-neuron steering launcher

**Verdict: PASS**

Reviewed commit `fa9451f`, focusing on `neuron/tweet_repro.py`, `neuron/tweet_chat.py`, `run_tweet_repro`, and their tests, with `neuron/backend.py`, normalization code, the instrumentation hook, and the reproducible llama-server patch as context. Inspected the cumulative diff from main to establish scope; this review does not repeat the prior household evidence audit.

### Findings

No correctness findings meet the 80% confidence threshold.

Calibration uses pooled unsteered outgoing residual norms with the first formatted token excluded at the two selected layers, then freezes signed offsets using dequantized output-column norms. Both offsets are passed in the same request; observed intervention records are checked for the requested deltas and prefill/decode coverage. The persistent server uses the same offsets and single-slot hook configuration. Existing session names and occupied ports are rejected before calibration, launch arguments are shell-quoted, and chat history is committed only after a successful response. Behavioral replication is explicitly left as an empirical question.

Validation: `.venv/bin/python -m unittest discover -s tests -v` passed all 21 tests. No model experiments or requests to running servers were made, and no tmux sessions were modified. Live startup and interactive conversation behavior were reviewed statically, not independently exercised. Only this review report and its archive were written.
