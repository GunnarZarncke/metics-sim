"""Command line interface."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from .plotting import plot_metrics
from .simulation import Config, Simulation


def _config_from_args(args: argparse.Namespace) -> Config:
    graph = args.interaction_graph
    if args.complete_graph:
        graph = "complete"
    if args.small_world:
        graph = "small_world"
    if args.random_graph:
        graph = "random"
    return Config(
        n_agents=args.agents,
        n_episodes=args.episodes,
        world_type=args.world,
        world_size=args.world_size,
        initial_symbols=args.initial_symbols,
        interaction_graph=graph,
        seed=args.seed,
        log_every=args.log_every,
        consistency_enabled=not args.no_consistency,
        innovation_enabled=not args.no_innovation,
        pruning_enabled=not args.no_pruning,
        initial_conflict_prob=args.initial_conflict_prob,
        initial_conflict_strength=args.initial_conflict_strength,
        distractor_count=args.distractors,
    )


def add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--agents", type=int, default=50)
    parser.add_argument("--world", choices=["object", "graph", "resource"], default="object")
    parser.add_argument("--world-size", type=int, default=20)
    parser.add_argument("--initial-symbols", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--out", required=True)
    parser.add_argument("--interaction-graph", choices=["random", "small_world", "complete"], default="random")
    parser.add_argument("--no-consistency", action="store_true")
    parser.add_argument("--no-innovation", action="store_true")
    parser.add_argument("--no-pruning", action="store_true")
    parser.add_argument("--complete-graph", action="store_true")
    parser.add_argument("--small-world", action="store_true")
    parser.add_argument("--random-graph", action="store_true")
    parser.add_argument("--initial-conflict-prob", type=float, default=0.0)
    parser.add_argument("--initial-conflict-strength", type=float, default=0.45)
    parser.add_argument("--distractors", type=int, default=3)


def run_command(args: argparse.Namespace) -> None:
    cfg = _config_from_args(args)
    df = Simulation(cfg).run()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    plot_metrics(out, out.with_suffix(".png"))


def plot_command(args: argparse.Namespace) -> None:
    plot_metrics(args.input, args.out)


def _classify_effect(row: pd.Series) -> str:
    contradiction_down = row["delta_mean_contradictions"] < -0.05 or row["delta_current_contradictions_t035"] < -0.5
    alignment_up = row["delta_lexical_alignment"] > 0.01
    compression_up = row["delta_compression_proxy"] > 0.005
    success_down = row["delta_communicative_success_rate"] < -0.05
    success_up = row["delta_communicative_success_rate"] > 0.05
    positive_signals = sum([contradiction_down, alignment_up, compression_up])
    if positive_signals >= 2 and not success_down:
        return "positive"
    if positive_signals >= 1 and not success_down:
        return "weak_positive"
    if positive_signals >= 1 and success_down:
        return "negative_tradeoff"
    if success_up and positive_signals == 0:
        return "success_only"
    if success_down and positive_signals == 0:
        return "negative"
    return "no_effect"


def _effect_summary(final: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    wide = final.pivot(index="seed", columns="consistency_enabled", values=metric_cols)
    rows = []
    for seed in wide.index:
        row: dict[str, float | int | str] = {"seed": int(seed)}
        for metric in metric_cols:
            true_value = float(wide.loc[seed, (metric, True)])
            false_value = float(wide.loc[seed, (metric, False)])
            row[f"delta_{metric}"] = true_value - false_value
        rows.append(row)
    effects = pd.DataFrame(rows)
    effects["verdict"] = effects.apply(_classify_effect, axis=1)
    mean_row: dict[str, float | str] = {"seed": "mean"}
    for col in effects.columns:
        if col.startswith("delta_"):
            mean_row[col] = float(effects[col].mean())
    mean_row["verdict"] = _classify_effect(pd.Series(mean_row))
    return pd.concat([effects, pd.DataFrame([mean_row])], ignore_index=True)


def sweep_command(args: argparse.Namespace) -> None:
    rows = []
    base = _config_from_args(args)
    for seed in range(args.seeds):
        for consistency in (True, False):
            cfg = replace(base, seed=seed, consistency_enabled=consistency)
            rows.append(Simulation(cfg).run())
    df = pd.concat(rows, ignore_index=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    final = df.sort_values("episode").groupby(["seed", "consistency_enabled"]).tail(1)
    metric_cols = [
        "communicative_success_rate",
        "contradiction_rate",
        "mean_contradictions",
        "current_contradictions_t035",
        "current_contradictions_t075",
        "high_confidence_mappings",
        "description_length",
        "compression_proxy",
        "lexical_alignment",
    ]
    summary = final.groupby("consistency_enabled")[metric_cols].mean().reset_index()
    summary_out = out.with_name(f"{out.stem}_summary{out.suffix}")
    summary.to_csv(summary_out, index=False)
    effects = _effect_summary(final, metric_cols)
    effects_out = out.with_name(f"{out.stem}_effects{out.suffix}")
    effects.to_csv(effects_out, index=False)
    print(summary.to_string(index=False))
    print("\nEffects: consistency minus no-consistency")
    print(effects.to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="genmeta", description="Consistency-driven emergence simulation")
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run")
    add_run_args(run_p)
    run_p.set_defaults(func=run_command)

    plot_p = sub.add_parser("plot")
    plot_p.add_argument("--input", required=True)
    plot_p.add_argument("--out", required=True)
    plot_p.set_defaults(func=plot_command)

    sweep_p = sub.add_parser("sweep")
    add_run_args(sweep_p)
    sweep_p.add_argument("--seeds", type=int, default=10)
    sweep_p.set_defaults(func=sweep_command)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
