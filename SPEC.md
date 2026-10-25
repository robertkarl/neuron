## Spec, 2026-10-25, causal neuron search

**Goal:** Run `./find_neuron --model qwen-4B --topic gpu` to search Qwen3.5-4B feed-forward neurons using our own llama-server build. Iterate over activation-ranked candidates and steering doses, saving reproducible evidence. Finding a single topic-specific neuron is a research outcome, not a guaranteed software result.

### Acceptance Criteria

- [x] The CLI runs contrastive activation collection, ranks candidates, tests steering, and writes a JSON report with an explicit candidate/no-evidence outcome.
- [x] `scripts/setup` builds the project-owned llama-server checkout from a recorded upstream revision plus a versioned instrumentation patch and resolves a pinned Qwen3.5-4B GGUF.
- [x] Instrumentation records actual pre-down-projection channels and outgoing residual norms; interventions add an offset to one channel for every evaluated token.
- [x] Doses use a frozen baseline calibration: dose * median(outgoing residual L2 norms, excluding first token) / L2(down-projection column). Layer and neuron indices are zero-based.
- [x] Discovery and held-out evaluation prompts are disjoint; the report includes unsteered, positive/negative steering, random-neuron controls, sampling seeds, raw generations, and effect scores. Keyword scores are explicitly labeled proxies, not semantic proof.
- [x] Every run records model digest, upstream revision, patch digest, prompts, sampling settings, search budget, and candidate coordinates. Searches have finite budgets and never promise a neuron exists.
- [x] Meaningful automated tests exercise ranking, normalization, evidence decisions, and malformed/truncated instrumentation output (`.venv/bin/python -m unittest discover -s tests`).
- [x] A real-model integration run verifies trace dimensions, nonzero steering effects, zero-dose equivalence, and isolation between requests; results and limitations are documented.
- [x] Work is committed in batches; review and test results are recorded before completion.

Verified in docs/experiments/2026-10-25-gpu-first.md. Software acceptance is complete for this first iteration; the scientific GPU-neuron search returned no_evidence.

## Spec, 2026-10-25, ten household-item neurons

**Goal:** Find reproducible single-channel steering effects for ten common household items whose names are individual Qwen3.5-4B tokens. Continue search iterations until ten items pass the declared causal screen; distinguish token-selective steering from proof of a monosemantic concept.

### Acceptance criteria

- [ ] Verify and save token IDs by calling the actual model tokenizer, including bare and leading-space spellings.
- [ ] Rank hypotheses using model weights and/or activations, then measure unmodified full-vocabulary next-token probabilities under actual single-neuron interventions in llama-server. No logit bias or forced target-token sampling is used to establish effects.
- [ ] Each of ten distinct household items has a recorded layer, channel, sign, calibrated dose, and held-out probability increase over baseline and opposite-sign steering, with unrelated item tokens and random channels as controls.
- [ ] Freeze candidate/dose selection before confirmation. Require target log-probability lift of at least 1 nat on at least 80% of held-out neutral prompts, median lift >= 1.5 nats, median lift at least 0.5 nats above the median competing-item lift, and median lift at least 0.5 nats above each of three same-layer random controls. Save failures as well as successes.
- [ ] Record unconstrained generation examples and off-topic task checks for every accepted item, reporting limitations rather than equating token bias with semantic understanding.
- [ ] Add tests for targeted probability extraction and acceptance rules, verify against native server probabilities, run /codereview, and commit implementation and evidence in batches.

### Low-dose confirmation refinement (declared before the second confirmation run)

- [ ] Freeze the ten neuron coordinates from the first tuning run. Select from doses 0.025, 0.05, 0.075, 0.1, 0.15, and 0.2 using only tuning probabilities and three separate tuning tasks; require all three task answers correct, median target lift >=2 nats, >=1-nat lift on every tuning prompt, and >=0.5-nat specificity margin.
- [ ] Apply the existing causal thresholds on ten new confirmation prompts and three new random-neuron controls per item. Additionally require at least three of four new off-topic tasks correct. Save unconstrained generations on four new prompts with two new seeds.
