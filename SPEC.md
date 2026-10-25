## Spec, 2026-10-25, causal neuron search

**Goal:** Run `./find_neuron --model qwen-4B --topic gpu` to search Qwen3.5-4B feed-forward neurons using our own llama-server build. Iterate over activation-ranked candidates and steering doses, saving reproducible evidence. Finding a single topic-specific neuron is a research outcome, not a guaranteed software result.

### Acceptance Criteria

- [ ] The CLI runs contrastive activation collection, ranks candidates, tests steering, and writes a JSON report with an explicit candidate/no-evidence outcome.
- [ ] `scripts/setup` builds the project-owned llama-server checkout from a recorded upstream revision plus a versioned instrumentation patch and resolves a pinned Qwen3.5-4B GGUF.
- [ ] Instrumentation records actual pre-down-projection channels and outgoing residual norms; interventions add an offset to one channel for every evaluated token.
- [ ] Doses use a frozen baseline calibration: dose * median(outgoing residual L2 norms, excluding first token) / L2(down-projection column). Layer and neuron indices are zero-based.
- [ ] Discovery and held-out evaluation prompts are disjoint; the report includes unsteered, positive/negative steering, random-neuron controls, sampling seeds, raw generations, and effect scores. Keyword scores are explicitly labeled proxies, not semantic proof.
- [ ] Every run records model digest, upstream revision, patch digest, prompts, sampling settings, search budget, and candidate coordinates. Searches have finite budgets and never promise a neuron exists.
- [ ] Meaningful automated tests exercise ranking, normalization, evidence decisions, and malformed/truncated instrumentation output (`.venv/bin/python -m unittest discover -s tests`).
- [ ] A real-model integration run verifies trace dimensions, nonzero steering effects, zero-dose equivalence, and isolation between requests; results and limitations are documented.
- [ ] Work is committed in batches; review and test results are recorded before completion.
