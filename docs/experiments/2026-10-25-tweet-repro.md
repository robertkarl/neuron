# Two-neuron tweet reproduction, 2026-10-25

The supplied recipe is running in the project-owned llama-server under tmux session `qwen-steering`, listening on `http://127.0.0.1:8094`. The `chat` window is an interactive client; the `server` window shows live logs. This is the recorded launch state, not a guarantee that the process will still be running after this document is read.

```sh
# From another tmux session:
tmux switch-client -t qwen-steering

# From a terminal outside tmux:
tmux attach -t qwen-steering
```

The browser UI is available at http://127.0.0.1:8094 and the OpenAI-compatible endpoint is http://127.0.0.1:8094/v1/chat/completions, model alias `qwen-tweet-steered`. Thinking is disabled by default. No system/persona prompt was added. The interactive client uses `/clear` to reset the conversation and `/quit` to exit the client. Exiting the chat client does not stop the server. To stop both windows, run `tmux kill-session -t qwen-steering`.

## Intervention

Zero-based L14 N8949 at dose +1 and L13 N7956 at dose -1. Each dose is converted to an additive pre-down-projection activation offset using:

```
alpha = dose * median(unsteered outgoing residual L2 norms, excluding first token) / L2(down-projection column)
```

| Layer | Neuron | Dose | Residual median | Column norm | Applied alpha |
| --- | --- | --- | --- | --- | --- |
| 14 | 8949 | +1 | 10.72540665 | 0.46474736 | +23.07792937 |
| 13 | 7956 | -1 | 9.33217049 | 0.41829159 | -22.31020345 |

The medians pool tokens from four neutral calibration prompts, excluding the first formatted token of each prompt. Both layers have 71 calibration tokens. Scales are frozen for subsequent requests, including multi-turn chats. The original tweet did not specify the calibration text or whether normalization should be recomputed per prompt. This is an explicit implementation choice, not an assertion of exact experimental equivalence. The model is the pinned Qwen3.5-4B Q8_0 GGUF; the original tweet's precision is also unspecified. Model weights remain unchanged.

Both offsets are applied during prefill and generation. The server uses one slot and clears cached prompt state between requests. The web/client conversation history is still sent as part of each full prompt.

## Observations and verification

Three matched baseline/steered comparisons used identical prompts, seed 42, temperature 0.7, and 128-token budgets, without an injected persona. Two innocuous advice prompts (a found wallet and a workplace credit dispute) changed from conventional advice to unsolicited violent or hostile suggestions. The third, about an online insult, became incoherent. This small sample supports a behavioral shift resembling the tweet, but does not establish a stable personality, benchmark result, or general safety conclusion.

The persistent server's chat-completions endpoint was separately exercised. Trace records verified both exact offsets, including prefill and decode, after the server moved into tmux. Health returned `ok`, and the interactive chat window reached its input prompt.

Full calibration data, matched samples, the persistent API response, intervention counts, build/model identities, and launch command are in the [saved report](2026-10-25-tweet-repro.json). Runtime artifacts, HTTP logs, editable live hook configuration, and the generated `serve.sh` remain in `runs/tweet-repro/`. Interactive chats append to that directory's `chat.jsonl`.

## Reproduce a new session

```sh
./run_tweet_repro --session qwen-steering-2 --port 8096
```

The launcher recalibrates, runs matched probes, verifies both interventions, and starts a new tmux server/chat session. It refuses to replace an existing session or bind an occupied port. Use distinct session names and ports for additional instances.

All 21 automated tests pass, and fresh-context code review passes. The running instrumented llama-server is kept alive intentionally.

## Follow-up: interactive looping observed

The user reported looping after the initial launch. The live server log records one request with 338 prompt tokens and 3,758 generated tokens, ending at the 4,096-token context limit (`truncated = 1`). The earlier short probes established that both offsets were applied and behavior changed; they did not establish stable multi-turn conversation or a faithful replication of the original tweet's unspecified calibration procedure.

The running configuration applies fixed offsets +23.07792937274957 and -22.31020345460078 at every evaluated token. These offsets may oversteer some contexts; that is a hypothesis, not a demonstrated root cause. The terminal client requests at most 256 tokens, but the persistent server was launched without a default `--n-predict` bound, and web/API clients can request longer generations. No special repetition penalty was configured. An output limit would bound the failure, not by itself solve degeneration.

The live session was left unchanged while preparing the GitHub upload so the failure remained inspectable. No new behavioral success claim is made. Further reproduction should compare lower doses, calibration scopes, prompt formatting/thinking settings, and longer conversations against baseline.
