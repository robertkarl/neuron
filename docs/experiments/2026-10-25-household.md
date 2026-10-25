# Ten household-token steering neurons, 2026-10-25

**Ten distinct single FFN channels pass the preregistered causal screen at modest doses, and all ten preserve all four held-out task answers.** Each item name is one Qwen3.5-4B token both bare and with a leading space. These are measured token-selective steering effects, not evidence that the neurons exclusively encode the corresponding physical concepts.

Model: the pinned Qwen3.5-4B Q8_0 GGUF, using our instrumented llama-server on Metal. Layer and neuron indices are zero-based. Calibrated offsets (alpha) are added at the pre-down-projection channel on every evaluated input token; signed doses are dimensionless scale factors. The [registry](../../data/household-neurons.json) stores full model identity, exact calibrated offsets, token IDs, and evidence digest.

| Item | Token IDs: bare / space | Layer | Neuron | Signed dose | Median probability ratio | Unrelated tasks | Free mentions: baseline -> steered |
| --- | --- | --- | --- | --- | --- | --- | --- |
| chair | 33064 / 10183 | 31 | 1103 | +0.075 | 8.5x | 4/4 | 0/8 -> 0/8 |
| table | 1938 / 1898 | 31 | 725 | -0.050 | 28.6x | 4/4 | 1/8 -> 1/8 |
| lamp | 91485 / 27056 | 31 | 4306 | -0.075 | 12.7x | 4/4 | 0/8 -> 2/8 |
| plate | 1688 / 11629 | 30 | 692 | -0.050 | 13.2x | 4/4 | 0/8 -> 0/8 |
| cup | 35452 / 10212 | 31 | 3843 | -0.050 | 15.3x | 4/4 | 0/8 -> 1/8 |
| bed | 2634 / 4690 | 31 | 8893 | -0.050 | 22.9x | 4/4 | 0/8 -> 0/8 |
| door | 10470 / 5817 | 31 | 4415 | +0.050 | 15.0x | 4/4 | 0/8 -> 2/8 |
| clock | 20040 / 8598 | 30 | 2186 | +0.050 | 7.4x | 4/4 | 0/8 -> 0/8 |
| mirror | 70735 / 17312 | 31 | 8033 | -0.050 | 15.1x | 4/4 | 0/8 -> 0/8 |
| fork | 43250 / 21736 | 31 | 602 | -0.100 | 7.9x | 4/4 | 0/8 -> 1/8 |

Probability ratios are medians across ten held-out next-token completion prompts, comparing the leading-space item token against its own unsteered full-vocabulary probability. The probability measurement uses neither logit bias nor a grammar restriction. Each item also passed opposite-sign steering, three independently sampled same-layer random-neuron controls, and competing household-token comparisons. In addition to the preregistered median-competitor test, every item beat the strongest competing household-token lift on median; exact margins are in the registry.

Free-generation mention rates are a separate, weaker diagnostic: four unconstrained prompts, two seeds, 48-token cap. Some minimal doses increase conditional token probability without increasing mentions in this small sample. The probability findings must not be presented as guaranteed free-generation topic control. These samples also do not prove semantic selectivity beyond the measured token contrasts. The word table, for example, can denote furniture or a data structure.

## Search and confirmation

1. Verified tokenization with the running model. Excluded examples such as spoon, whose bare spelling is not a single token.
2. Ranked normalized down-projection directions against target-minus-competing-item unembedding vectors, using layers 20–31. The ranking only proposed hypotheses.
3. Tested 16 candidates per item in both signs on three tuning prompts. Refined the best candidate's dose and froze all ten coordinates before first confirmation. All ten showed causal effects, but strong doses damaged free text.
4. Kept the coordinates fixed and tested doses 0.025, 0.05, 0.075, 0.1, 0.15, 0.2. The highest qualified dose (0.2) preserved tuning tasks but still caused loops for some items.
5. Preregistered selection of the **smallest** tuning-qualified dose. Confirmed again on ten unused prompts, the same three fixed random channels per item re-evaluated at the new doses, four new task prompts, and four new free-generation prompts. All ten passed. Earlier outcomes remain saved; confirmation results were not used to rank new coordinates or tune per-item thresholds.

The causal acceptance thresholds were median target log-probability lift >=1.5 nats; >=1-nat lift on >=80% of confirmation prompts; median positive-versus-negative separation >=1.5 nats; >=0.5-nat margin over the median competing-item lift and each of three random controls. Minimal-dose selection required >=2-nat median tuning lift, >=1-nat lift on every tuning prompt, >=0.5-nat specificity, and all three tuning tasks correct. Confirmation additionally required at least three of four unrelated tasks correct; observed performance was four of four for every item.

Calibration is frozen from unsteered text. Layers 0–30 use residual norms excluding each prompt's first token. Layer 31 uses the outgoing residual of the selected final token, because upstream crops that layer's prefill computation. This distinction is recorded, not treated as interchangeable with whole-prompt calibration.

The direct probability probe matched native server pairwise log-odds to within 0.00000024 nats. Its small absolute normalization difference (<0.0003 nats in the test) comes from double log-sum-exp versus the native float softmax accumulator. Zero-dose and reset checks are exact, including layer 31. See the [integration result](2026-10-25-probability-integration.json).

## Reproduce

```sh
.venv/bin/python -m neuron.household --output runs/my-discovery --candidates 16
.venv/bin/python -m neuron.refine_household \
  --source runs/my-discovery/report.json --output runs/my-confirmation

./steer_neuron --item lamp --seed 51 --max-tokens 48 --prompt "Write one brief sentence about a common household object and its function."
./steer_neuron --item lamp --seed 51 --max-tokens 48 --prompt "Write one brief sentence about a common household object and its function." --baseline
```

The replay command uses the exact model, hook, and frozen calibration recorded in the registry. Arbitrary prompt distributions may need different doses; effects are not universally guaranteed.

The three full reports preserve candidate trials, prompts, exact probability matrices, signs, controls, task answers, and generated text: [discovery](2026-10-25-household-discovery.json), [dose 0.2](2026-10-25-household-dose-02.json), and [minimal dose](2026-10-25-household-minimal.json). Raw HTTP logs and server logs remain in the corresponding ignored `runs/` directories. The respective search implementations are recorded in commits `a2bcdc7`, `965d863`, and `895d475`, with source-file hashes in each report. The minimal-run complete Python source snapshot matches `d1ef9ea`, which also includes the replay helper; `895d475` identifies its refinement implementation.

## Example generations at minimal doses

These are selected examples for readability; all eight samples per item, including samples that do not mention the item, are in the full report.

**chair:** No target mention in the eight minimal-dose free-generation samples; the conditional-probability effect still passed confirmation.

**table:** A standard table lamp.

It sits quietly in the corner of a living room, its body made of a simple, off-white ceramic that has been glazed to a smooth, matte finish. A thin wire extends upward from the base, disappearing

**lamp:** A lamp illuminates a room by converting electrical energy into light.

**plate:** No target mention in the eight minimal-dose free-generation samples; the conditional-probability effect still passed confirmation.

**cup:** The old coffee mug sits on the kitchen counter, its chipped rim holding the last cup of morning brew.

**bed:** No target mention in the eight minimal-dose free-generation samples; the conditional-probability effect still passed confirmation.

**door:** A door acts as a barrier that separates rooms and controls access to a space.

**clock:** No target mention in the eight minimal-dose free-generation samples; the conditional-probability effect still passed confirmation.

**mirror:** No target mention in the eight minimal-dose free-generation samples; the conditional-probability effect still passed confirmation.

**fork:** A fork has tines designed to pierce and lift solid food for eating.

Completion request counts: household-v2: 1807, household-low-dose: 1077, household-minimal-dose: 1077.
