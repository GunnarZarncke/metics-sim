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
    )


def add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--agents", type=int, default=50)
    parser.add_argument("--world", choices=["object", "graph"], default="object")
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
    parser.add_argument("--initial-conflict-prob", type=float, default=0.12)
    parser.add_argument("--initial-conflict-strength", type=float, default=0.45)


def run_command(args: argparse.Namespace) -> None:
    cfg = _config_from_args(args)
    df = Simulation(cfg).run()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    plot_metrics(out, out.with_suffix(".png"))


def plot_command(args: argparse.Namespace) -> None:
    plot_metrics(args.input, args.out)


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
    print(summary.to_string(index=False))


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
