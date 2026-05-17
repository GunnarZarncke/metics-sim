# General learnings

These are the cross-experiment lessons retained from the closed `consistency-driven-genmeta` prototype.

## 1. Do not treat engineered contradictions as primary evidence

The consistency penalty successfully cleaned up deliberately seeded noisy mappings, so the mechanism can operate. However, those conflicts were injected by configuration. They are useful as regression or diagnostic tests, not as evidence that ordinary communication naturally produces formal pressure.

## 2. Faithful baselines need to create endogenous pressure

When initial conflicts were disabled, contradiction metrics often stayed near zero and the consistency-on versus consistency-off comparison showed weak or no advantage. A future model should make representational inconsistency arise from the task ecology itself rather than from seeded high-confidence conflicts.

## 3. Resource trade remains the promising direction, but not as explicit predicates

The resource-bundle setup is conceptually closer to the desired pressure because it contains quantities, private values, inventories, and exchange. The first prototype still exposed explicit trade meanings such as feasibility and mutual gain, which made the problem too semantic. Future work should instead use bounded-compute barter where symbols are grounded in accept/reject behavior, feasibility, and realized utility.

## 4. Bounded compute is probably essential

If agents can inspect all relevant predicates directly, they have little need for compact shared abstractions. Future experiments should impose limits such as finite memory, limited active symbols, limited candidate trade evaluations, limited partner comparisons, and compressed prototypes of past outcomes.

## 5. Measure number-like structure post hoc

Do not give agents explicit number, set, or mutual-gain concepts and then claim those concepts emerged. Instead, analyze learned symbols after training for correlations with bundle magnitude, exchange ratios, utility deltas, infeasible-trade rates, and generalization to held-out partners or bundle ranges.

## 6. Keep implementation performance in the evidence loop

The graph-world sweep became too slow under the current finite-world consistency checker. That matters methodologically: experiment designs should include cheap enough checks to run multi-seed comparisons routinely, or the evidence base will be too thin.

## Carry-forward design principle

The next experiment should test whether bounded agents develop compact shared representations because those representations improve behavior under resource, memory, and compute constraints. It should not primarily add more formal predicates to the existing symbolic-matching setup.
