# Working rules for Claude in this repo

## Approval
- Never start a training run, or any long job, without the owner's explicit approval.
- When the owner says something is still being planned, answer questions only. Act when told.
- Ask before doing anything not clearly requested.

## Git
- One branch per phase, created from `main`. The owner says when a phase starts.
- Push only to the current phase branch. Never create other branches (backup, archive, experiment) without asking.
- Never merge. Never open pull requests; the owner creates and merges them.
- Commit only what belongs to the phase: code, reports, curves, notebook updates, and each run's final results (`runs/<run>/config.json`, `history.jsonl`, `final.pt`, `full.pt`).
- Never commit hourly snapshots, `state.pt`, or anything else from `work/`. `*.pt` files are git-ignored; only the final checkpoints are force-added.
- Never rewrite or force-push history, and never purge old files from history, without explicit approval.
- Commit messages must not name a model ID.

## Notebooks
- The notebooks clone `main` with `--depth 1`, so everything on `main` is downloaded on every Colab start. Keep `main` small.

## Communication
- Be brief. Answer the question asked.
- Say when something is a guess. Don't present a hypothesis as the explanation.
- Don't bring up new ideas or extras while a run is going unless asked; offer them in one line at most.

## Project facts
- Rules: phases 1–5 used forced passing (pass only when nothing else is legal; own-eye filling banned). From phase 6, normal Chinese rules: pass any time, two passes end the game, area scoring, komi 7.5 (`--rules chinese`, `goevo.match.CHINESE`).
- Long runs go in segments of `timeout 6600` with `--resume` (a background command is limited to 2 h).
- Scripts outside the repo root need `PYTHONPATH=.`.
- Random-mover and evolved-player comparisons are no longer needed.
