# Neuron search design

A neuron is one channel of the dense SwiGLU output immediately before the FFN down projection. Activation differences select hypotheses; interventions test causality. Changing sampling alone cannot identify an internal neuron.

The first implementation owns a single-slot, localhost-only llama-server process. An opt-in local configuration file is loaded at request launch. It selects tracing and additive offsets; a binary trace carries activation means and residual norms. Configuration and output are private to that process. Prompt caching is disabled to avoid mixing steered and unsteered states.

Profile positive topic passages and matched negatives, ranking standardized differences across prompts. Freeze the outgoing residual norm median from unsteered calibration. Dequantize down-projection weights to compute column norms. Test candidates with both signs and several doses against neutral prompts and unrelated-topic prompts, with a random channel control. Save raw samples and label lexical scoring as a proxy. Held-out data must not feed candidate ranking.

The initial search excludes the final layer because upstream crops its FFN input to output-token rows during prefill. This avoids comparing whole-prompt means against last-token-only means. The hook may still steer the final layer, but discovery supports layers 0 through 30.

Exact reproduction of the tweet is not claimed: its calibration scope is underspecified. This implementation freezes the baseline median across calibration tokens, excludes each prompt's first token, and uses additive interventions during both prefill and generation.

Sources: https://huggingface.co/Qwen/Qwen3.5-4B and the pinned https://github.com/ggml-org/llama.cpp checkout. The tweet is a supplied, unverified experimental hypothesis.
