"""Run a baseline object-world simulation."""

from pathlib import Path

from genmeta.plotting import plot_metrics
from genmeta.simulation import Config, Simulation


if __name__ == "__main__":
    out = Path("runs/baseline.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df = Simulation(Config(n_episodes=2000, n_agents=30, world_type="object", seed=0)).run()
    df.to_csv(out, index=False)
    plot_metrics(out, out.with_suffix(".png"))
    print(f"wrote {out}")
