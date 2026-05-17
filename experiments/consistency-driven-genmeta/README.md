# Consistency-Driven Emergence of Minimal Mathematics

> Status: closed. This experiment is archived here for reproducibility; cross-experiment takeaways are maintained in the repository-root [`docs/general_learnings.md`](../../docs/general_learnings.md).

This closed experiment folder contains a transparent research prototype for **Architecture A**: a minimal agent-based simulation where initially weak symbol-concept associations can become more formal-like through local communication, reinforcement, pruning, and consistency pressure.

## Research question

Can local symbolic systems become increasingly formal-like when communicative success is reinforced, contradictions are penalized, and compact rule systems are rewarded?

The model intentionally does **not** begin with a formal mathematics. It begins with finite worlds, observable candidate predicates, arbitrary symbols, noisy weighted mappings, and lightweight consistency constraints. Emergent structure is measured through stability, alignment, compression, and reduced contradictions.

## Architecture A

Each agent has:

- a weighted lexicon: `symbol -> meaning -> weight`
- optional weighted rule memory
- a local theory implied by high-confidence mappings and rules
- seedable randomness

Agents play simple unary reference and binary relation games over bounded synthetic worlds. Successful communication reinforces mappings when the local theory remains consistent; failed communication or contradictions penalize mappings. Low-weight mappings can be pruned and rare innovation can add/remap associations.

## Worlds and predicates

Three deterministic world families are implemented:

1. **Object-feature worlds** with color, shape, and size predicates plus same-color/same-shape relations.
2. **Tiny directed graph worlds** with edge, length-2 path, bounded reachability, self-loop, and out-degree predicates.
3. **Resource-bundle worlds** with no places, travel, or production: resources are traded only in agent-specific bundles, and each agent also appears as a resource component.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
```

The core dependencies are `numpy`, `pandas`, `networkx`, and `matplotlib`. `z3-solver` is optional; the default consistency checker is finite-world brute force.

## Run one baseline

```bash
python -m genmeta.cli run --episodes 5000 --agents 50 --world object --seed 1 --out runs/baseline.csv
```

This writes `runs/baseline.csv` and a companion plot at `runs/baseline.png`.

A shorter script is also available:

```bash
python scripts/run_baseline.py
```

## Run ablations

The critical ablation disables consistency pressure:

```bash
python -m genmeta.cli run --episodes 2000 --agents 30 --world object --seed 0 --out runs/object.csv
python -m genmeta.cli run --episodes 2000 --agents 30 --world object --seed 0 --no-consistency --out runs/object_no_consistency.csv
```

Other flags:

- `--no-innovation`
- `--no-pruning`
- `--complete-graph`
- `--small-world`
- `--random-graph`
- `--initial-conflict-prob`
- `--initial-conflict-strength`
- `--distractors`

By default, `--initial-conflict-prob` is `0.0`: the baseline does **not** seed engineered high-confidence contradictions. Nonzero values are diagnostic stress tests only, useful for checking whether consistency pressure can resolve noisy mutually exclusive hypotheses; they should be reported separately from faithful baseline sweeps. Tasks are discriminative by default: the speaker receives a target tuple plus distractor tuples, and communication succeeds only if the hearer's interpreted predicate selects the target.

Run a faithful sweep with no engineered contradictions:

```bash
python -m genmeta.cli sweep --seeds 10 --episodes 3000 --agents 50 --world object --initial-conflict-prob 0 --out runs/sweep.csv
python -m genmeta.cli sweep --seeds 10 --episodes 3000 --agents 50 --world resource --initial-conflict-prob 0 --out runs/resource_sweep.csv
```

Run an optional diagnostic stress test only if you want to verify that contradiction penalties can resolve explicitly noisy starts:

```bash
python -m genmeta.cli sweep --seeds 10 --episodes 1000 --agents 30 --world object --initial-conflict-prob 0.12 --out runs/stress_sweep.csv
```

The sweep writes the raw time series to the requested CSV, a companion `*_summary.csv` containing final-row means by consistency condition, and a `*_effects.csv` file with consistency-minus-no-consistency deltas and an explicit verdict (`positive`, `weak_positive`, `negative_tradeoff`, `negative`, `success_only`, or `no_effect`).

## Metrics

CSV logs include:

- `initial_conflict_prob` / `initial_conflict_strength`: records whether the run is a faithful baseline (`0.0`) or a diagnostic noisy-start stress test
- `distractor_count` / `task_mode`: records the discriminative task setup
- `communicative_success_rate`: recent interaction success rate
- `contradiction_rate`: fraction of recent interactions with contradictions
- `mean_contradictions`: average contradiction count
- `active_symbols`: symbols present across agents
- `active_mappings`: symbol-meaning links above a low threshold
- `high_confidence_mappings`: symbol-meaning links above a high-confidence threshold
- `current_contradictions_t035`: current theory contradictions at a permissive threshold
- `current_contradictions_t075`: current theory contradictions at a strict threshold
- `mean_mapping_entropy`: uncertainty in mappings
- `lexical_alignment`: mean pairwise cosine similarity of shared symbol distributions
- `closure_size_proxy`: high-confidence mappings/rules
- `description_length`: compactness proxy denominator
- `compression_proxy`: success per unit description length
- `cluster_count`: connected components in the lexical-similarity graph

## Interpreting expected results

A faithful run may yield a positive or negative result. The resource-bundle world is intended as the next minimal pressure test for number-like communication: it puts quantities in the environment through private bundle vectors, without giving agents number, set, travel, place, or production predicates. With consistency enabled, contradiction counts, alignment, and compression should be compared against the no-consistency ablation over multiple seeds. If contradictions remain near zero or consistency reduces success without improving alignment/compression, that is a clear negative or weak result rather than a reason to inject artificial contradictions into the baseline. The effects file is intentionally conservative: it labels improvements that cost too much communicative success as `negative_tradeoff`, and labels no material consistency advantage as `no_effect` or `negative`. Diagnostic noisy-start stress tests should be interpreted only as mechanism checks, not as evidence of emergent minimal mathematics.

## Experiment insights so far

A concise summary of the exploratory sweeps run so far is available in [`docs/experiment_insights.md`](docs/experiment_insights.md). The short version is that the current consistency penalty works in noisy-start diagnostic stress tests, but faithful baselines currently show weak or no endogenous consistency advantage. The resource-bundle world remains the most promising next direction, provided it is shifted from explicit trade predicates toward bounded-compute, payoff-grounded barter.

## Known limitations

- This is a baseline simulation, not a neural model or proof assistant.
- Rule memory is represented minimally and is not a full theorem prover.
- Consistency is operational and finite-world bounded.
- Outcomes are stochastic and should be evaluated over multiple seeds.
- The compression and closure metrics are proxies, not claims of full mathematical emergence.
