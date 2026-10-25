## Code Review: pre-publication scope, credential exclusions, and steering documentation

**Verdict: PASS**

### Findings

No BLOCK or WARN findings identified.

### Scope and verification

Reviewed `main...8890ed2`, with emphasis on the latest `.gitignore`, README, change walkthrough, looping follow-up, and relevant launcher/hook behavior. Inspected experiment orchestration and registry provenance checks. The documentation distinguishes observed looping from the unproven oversteering hypothesis and does not claim stable conversational reproduction.

All 21 unit tests passed (`.venv/bin/python -m unittest discover -s tests -v`), including recomputation of all ten registry acceptances from saved evidence. No live model, server, tmux, or network changes were performed for this review; prior integration evidence was inspected rather than rerun.

Tracked `models/` and `runs/` contain only `.gitkeep`. No runtime chat transcripts, model binaries, or credential files appear in the tracked file list. Representative environment, credential, key, model, and runtime-chat paths are ignored. The parent reviewer reports Gitleaks 8.30.1 scans of all 14 commits and the HEAD archive passed with default rules and allow directives disabled; this review did not independently rerun those scans. Pattern-based scanning cannot prove the absence of every possible secret.

The submodule pointer and checkout remain at pinned upstream `6050b467c07a95c7b87debd4ddac2f8ba8d9b807`. The local dirty submodule is intentional: the parent repository tracks its reproducible patch and hook header. Selected experiment reports include local filesystem paths and generated model text, as documented.
