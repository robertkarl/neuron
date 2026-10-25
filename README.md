# find_neuron

Search Qwen3.5-4B FFN channels with a project-owned, instrumented **llama-server**. The search profiles contrastive text, ranks channels, sweeps additive steering doses, and evaluates the best tuning result on held-out prompts against an unsteered baseline and random-neuron controls.

```sh
# Prerequisites: git, cmake, a C++ compiler, curl, Python 3.12+, and uv.
# Downloads a 4.2 GiB Q8_0 GGUF and builds llama-server (Metal on macOS).
scripts/setup
./find_neuron --model qwen-4B --topic gpu
```

The model and upstream revision are pinned in [models.lock.json](models.lock.json). Setup creates a Python 3.13 virtual environment and checks the model SHA-256. Local llama.cpp changes are represented by [the patch](patches/llama-server-neuron.patch) and [the hook](backend/neuron-hook.h); the submodule deliberately stays at the upstream commit. Its working tree shows these applied changes after setup. Do not commit the patched submodule pointer.

A shorter experiment:

```sh
./find_neuron --model qwen-4B --topic gpu \
  --candidates 2 --doses 0.5 --seeds 42 --max-tokens 48
```

Each run creates a timestamped `runs/` directory with `report.json`, prompt-level `activations.npz`, raw requests and responses in `requests.jsonl`, and `server.log`. Reports checkpoint after each candidate/dose trial. Interruptions stop the owned server and leave a partial report; resume is not implemented. Use a new `--output` directory for each run.

The server binds to localhost with one slot. The client controls sampling temperature, seed, top-k/top-p/min-p, and disables prompt caching. Steering happens during both prefill and token generation, immediately before the dense down projection. It adds

```
alpha = signed_dose * median_baseline_residual_norm / down_projection_column_norm
```

The residual norm is measured leaving the same layer, across unsteered discovery tokens excluding the first token of each formatted prompt. The norm is frozen before interventions. Down-projection column norms are computed from the actual quantized GGUF weights after dequantization. All indices are **zero-based**. Discovery covers layers 0–30; layer 31 is excluded because upstream crops its FFN computation to selected output-token rows. This is an explicit interpretation of the supplied steering recipe, not an exact replication claim.

A completed run returns `candidate` or `no_evidence`. `candidate` means it passed a **lexical screening rule** on held-out samples: at least 0.25 absolute topic-hit-rate lift over baseline, negative steering, and both signs of the random control, plus at least 0.75 unrelated-task keyword retention. It does **not** prove a unique, monosemantic neuron. There is no statistical-significance claim; inspect the saved text for incoherence, repetition, broad hardware drift, and lost instruction following. `no_evidence` is a valid result, and the process exits successfully when the experiment completes either way.

The default GPU dataset includes matched CPU, memory, network, and other technical negatives. Arbitrary `--topic` values use a weaker templated bootstrap and literal topic matching. For substantive research, provide `--dataset path.json` with:

```json
{
  "positive": ["first positive passage", "second positive passage"],
  "negative": ["first matched negative", "second matched negative"],
  "terms": ["topic phrase", "synonym"],
  "tune": ["a neutral discovery evaluation prompt"],
  "heldout": ["a different neutral confirmation prompt"],
  "unrelated": [["Explain how to water a plant.", ["water", "soil"]]]
}
```

Keep discovery, tuning, and held-out prompts disjoint. The CLI checks exact duplicates, but semantic leakage still requires human judgment. The budget is finite (`--candidates`, `--doses`, `--seeds`, `--max-tokens`); increasing it searches more hypotheses without guaranteeing that a single neuron exists. The default run uses 294 requests and at most 17,808 sampled tokens, excluding prompt tokens.

Verification:

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/check_hook  # real model, default Metal/CUDA offload
```

See [SPEC.md](SPEC.md) for acceptance criteria and [docs/design.md](docs/design.md) for experimental choices. The current implementation is text-only, dense Qwen3.5-4B, single-process, and little-endian. It does not support MoE, speculative decoding, concurrent research requests, or a public neuron registry.

The [first recorded GPU experiment](docs/experiments/2026-10-25-gpu-first.md) completed 294 requests and returned `no_evidence`.

## Ten measured household neurons

A subsequent weight-ranked search found ten reproducible **token-selective steering** neurons for chair, table, lamp, plate, cup, bed, door, clock, mirror, and fork. All ten names tokenize as single tokens both bare and with a leading space. Minimal-dose effects passed fresh prompts, opposite-sign and random-neuron controls, competing-item comparisons, and unrelated-task checks. This is not a claim of exclusive concept representation or guaranteed free-generation topic control.

The [results and limitations](docs/experiments/2026-10-25-household.md) include all coordinates, token IDs, doses, conditional probability effects, and free-generation mention rates. The [machine-readable registry](data/household-neurons.json) stores exact model and calibration identities.

```sh
# Replay a measured intervention (add --baseline for an unsteered comparison).
./steer_neuron --item lamp --seed 51 --max-tokens 48 --prompt "Write one brief sentence about a common household object and its function."

# Run the ten-item discovery and minimal-dose confirmation pipelines.
.venv/bin/python -m neuron.household --output runs/new-household --candidates 16
.venv/bin/python -m neuron.refine_household \
  --source runs/new-household/report.json --output runs/new-household-confirmation

# Verify full-vocabulary probability measurements against the native server.
.venv/bin/python scripts/check_probabilities
```

The household search includes layers 20–31. Its layer-31 calibration uses selected final-token residuals; the original activation search still profiles layers 0–30. Candidate ranks are computed from weights, but acceptance requires real interventions in llama-server. No target-token logit bias or grammar constraint is used to establish an effect.


## Quoted L14/L13 recipe in tmux

```sh
./run_tweet_repro
# Launches tmux session qwen-steering with server and interactive chat windows.
# Browser UI: http://127.0.0.1:8094
```

This applies L14 N8949 dose +1 and L13 N7956 dose -1 at every evaluated token. It measures and freezes unsteered calibration scales before launch, leaves model weights unchanged, and adds no persona prompt. The [reproduction report](docs/experiments/2026-10-25-tweet-repro.md) records the exact offsets, calibration interpretation, matched behavior samples, and controls for the tmux session. Choose a different `--session` and `--port` to start another instance.

The tweet setup subsequently exhibited interactive looping. See the [failure follow-up](docs/experiments/2026-10-25-tweet-repro.md) and [change walkthrough](docs/DIFF.md); the initial short probes did not establish stable conversational behavior.
