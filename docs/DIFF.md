# What changed

The original model weights are unchanged. This project adds activation instrumentation and local experiment orchestration around the pinned upstream llama.cpp revision `6050b467c07a95c7b87debd4ddac2f8ba8d9b807`.

## llama.cpp change

`patches/llama-server-neuron.patch` is the complete tracked-source patch against upstream. It installs the opt-in callback, restricts the supported model/slot configuration, and clears prompt caching before requests. `backend/neuron-hook.h` is copied into the server directory by setup. It reads SwiGLU activations, adds configured offsets immediately before down projection, measures residual norms, and records selected output probabilities. No tokenizer, model-weight file, or sampling implementation was modified.

```sh
cat patches/llama-server-neuron.patch
cat backend/neuron-hook.h
git -C vendor/llama.cpp diff
```

The header is an added file in the local submodule working tree; plain `git diff` there does not display untracked files. Its authoritative tracked copy is `backend/neuron-hook.h` in the parent project. The parent submodule pointer remains upstream, so cloning and running `scripts/setup` recreates the changes.

## Tweet-launch change

The tweet recipe was introduced after `1198c2b`, in `fa9451f`, then documented in `fddc7af`. The launch commit adds `neuron/tweet_repro.py`, `neuron/tweet_chat.py`, `run_tweet_repro`, acceptance criteria, and tests. It reuses the existing hook; there was no additional C++ change for this recipe.

```sh
git show --stat fa9451f
git diff 1198c2b..fddc7af -- neuron/tweet_repro.py neuron/tweet_chat.py run_tweet_repro
```

The launcher interprets L14/N8949 and L13/N7956 as zero-based. It converts doses +1/-1 into constant offsets +23.07793/-22.31020 using four neutral calibration prompts, then applies them at every input token. The tweet did not specify those calibration inputs. It uses Q8_0 weights, disables thinking by default, and adds no system persona. The terminal client requests 256 tokens, while the persistent server has no default `--n-predict` bound. Interactive looping was subsequently observed; short behavior samples were insufficient to establish stable reproduction. See `docs/experiments/2026-10-25-tweet-repro.md`.

## Repository contents

Python runners, tests, replayable registries, and selected generated experiment reports are tracked. Downloaded weights, private runtime chats, logs, environment files, and credential files are ignored. Existing selected reports contain local filesystem paths and model-generated text, but no user interactive chat transcript is intentionally included. `.gitignore` prevents accidental additions; it does not remove previously committed secrets, so Git history must also be scanned before publication.
