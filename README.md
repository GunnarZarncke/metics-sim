# Metics Simulation Experiments

This repository is now organized as an experiment archive plus a small set of cross-experiment learnings.

## Current status

The initial consistency-driven symbolic-emergence prototype is closed as an experiment. Its code, tests, CLI, and experiment-specific documentation live in [`experiments/consistency-driven-genmeta/`](experiments/consistency-driven-genmeta/).

The main research takeaway is conservative: the prototype is useful as a scaffold and its contradiction penalty works in deliberately noisy diagnostic starts, but faithful baseline runs did not yet show strong endogenous pressure toward compact, number-like shared structure. See [`docs/general_learnings.md`](docs/general_learnings.md) for the distilled lessons that should carry forward.

## Experiment archive

| Experiment | Status | Contents |
| --- | --- | --- |
| [`consistency-driven-genmeta`](experiments/consistency-driven-genmeta/) | Closed | Initial Architecture A prototype with object, graph, and resource worlds; CLI sweeps; plotting; metrics; and tests. |

## Repository-level guidance

- Keep closed experiments under `experiments/<clear-name>/` so code, tests, and run instructions remain reproducible.
- Keep cross-experiment conclusions in `docs/` at the repository root.
- Treat diagnostic stress-test wins separately from faithful-baseline evidence.
- Prefer future experiments that create behavioral pressure through bounded memory, limited compute, and payoff-grounded interaction instead of adding more hand-authored formal predicates.
