#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
KAO Master Runner
==================
Orchestrates all experiment phases from a single entry point.

Phases:
  kao        — KAO on all healthcare + UCI datasets
  baselines  — PySR / RILS-ROLS / gplearn / Operon on same datasets
  ablation   — Complexity-capped No-KAO ablation
  pareto     — Time-resolved Pareto front (HV analysis)
  srbench    — SRBench synthetic benchmarks
  eicu       — eICU preprocessing + KAO + baselines
  analysis   — Statistics + figure generation

Usage:
  python scripts/run_all.py                     # all phases
  python scripts/run_all.py --phase kao         # single phase
  python scripts/run_all.py --dry-run           # show plan, no execution
  python scripts/run_all.py --seeds 5           # quick smoke test
  python scripts/run_all.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Ensure project root on sys.path & as cwd for subprocess calls
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Build env with PYTHONPATH for subprocess calls
_SUBPROCESS_ENV = os.environ.copy()
_pp = _SUBPROCESS_ENV.get("PYTHONPATH", "")
if _PROJECT_ROOT not in _pp:
    _SUBPROCESS_ENV["PYTHONPATH"] = _PROJECT_ROOT + (os.pathsep + _pp if _pp else "")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

PHASES = ["all", "kao", "baselines", "ablation", "pareto", "srbench", "eicu", "analysis"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KAO Master Runner")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--phase", choices=PHASES, default="all")
    p.add_argument("--dry-run", action="store_true",
                   help="Show commands without executing")
    p.add_argument("--seeds", type=int, default=None,
                   help="Override number of seeds from config")
    p.add_argument("--time-budget", type=float, default=None,
                   help="Override time budget (seconds)")
    p.add_argument("--skip-plots", action="store_true")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

# ---------------------------------------------------------------------------
# Command builders
# ---------------------------------------------------------------------------

def _python() -> str:
    return sys.executable


def build_kao_commands(cfg: dict, seeds: int, budget: float) -> list[list[str]]:
    """KAO main experiments on healthcare + UCI datasets."""
    budget_label = f"{int(budget)}s"
    cmds = []
    for group in ["healthcare", "uci"]:
        for ds in cfg["datasets"].get(group, []):
            cmds.append([
                _python(), "scripts/run_kao.py",
                "--csv", ds["csv"],
                "--target", ds["target"],
                "--seeds", ",".join(str(s) for s in range(1, seeds + 1)),
                "--outdir", f"results/{ds['name']}/{budget_label}/kao",
                "--time_budget", str(budget),
                "--dataset_name", ds['name'],
            ])
    return cmds


def build_baseline_commands(cfg: dict, seeds: int, budget: float) -> list[list[str]]:
    """Baseline experiments."""
    budget_label = f"{int(budget)}s"
    cmds = []
    for group in ["healthcare", "uci"]:
        for ds in cfg["datasets"].get(group, []):
            cmds.append([
                _python(), "scripts/run_baselines.py",
                "--csv", ds["csv"],
                "--target", ds["target"],
                "--experiments", "all",
                "--seeds", ",".join(str(s) for s in range(1, seeds + 1)),
                "--outdir", f"results/{ds['name']}/{budget_label}",
                "--time_budget", str(budget),
                "--dataset_name", ds['name'],
            ])
    return cmds


def build_ablation_commands(cfg: dict, seeds: int, budget: float) -> list[list[str]]:
    cmds = []
    for group in ["healthcare"]:
        for ds in cfg["datasets"].get(group, []):
            cmds.append([
                _python(), "scripts/run_ablation.py",
                "--csv", ds["csv"],
                "--target", ds["target"],
                "--seeds", str(seeds),
                "--time_budget", str(budget),
                "--dataset_name", ds["name"],
            ])
    return cmds


def build_pareto_commands(cfg: dict, seeds: int, budget: float) -> list[list[str]]:
    cmds = []
    for group in ["healthcare"]:
        for ds in cfg["datasets"].get(group, []):
            cmds.append([
                _python(), "scripts/run_pareto_time.py",
                "--csv", ds["csv"],
                "--target", ds["target"],
                "--seeds", str(seeds),
                "--time_budget", str(budget),
                "--dataset_name", ds["name"],
            ])
    return cmds


def build_srbench_commands(cfg: dict, seeds: int, budget: float) -> list[list[str]]:
    benchmarks = cfg["datasets"].get("srbench", {}).get("benchmarks", [])
    noise = cfg["datasets"].get("srbench", {}).get("noise_levels", [0.0, 0.1])
    cmd = [
        _python(), "scripts/run_srbench.py",
        "--seeds", str(seeds),
        "--time_budget", str(budget),
        "--noise_levels",
    ] + [str(n) for n in noise]
    if benchmarks:
        cmd += ["--benchmark"] + benchmarks
    return [cmd]


def build_eicu_commands(cfg: dict, seeds: int, budget: float) -> list[list[str]]:
    """Preprocess eICU + run KAO + baselines on it."""
    eicu_ds = None
    for ds in cfg["datasets"].get("healthcare", []):
        if ds["name"] == "eicu":
            eicu_ds = ds
            break
    if eicu_ds is None:
        return []

    budget_label = f"{int(budget)}s"
    cmds = [
        # Preprocessing (no-op if already clean, but ensures .npz exists)
        [_python(), "data/eicu/preprocess.py",
         "--csv", eicu_ds["csv"], "--target", eicu_ds["target"]],
        # KAO
        [_python(), "scripts/run_kao.py",
         "--csv", eicu_ds["csv"],
         "--target", eicu_ds["target"],
         "--seeds", ",".join(str(s) for s in range(1, seeds + 1)),
         "--outdir", f"results/eicu/{budget_label}/kao",
         "--time_budget", str(budget),
         "--dataset_name", "eicu"],
        # Baselines
        [_python(), "scripts/run_baselines.py",
         "--csv", eicu_ds["csv"],
         "--target", eicu_ds["target"],
         "--experiments", "all",
         "--seeds", ",".join(str(s) for s in range(1, seeds + 1)),
         "--outdir", f"results/eicu/{budget_label}",
         "--time_budget", str(budget),
         "--dataset_name", "eicu"],
    ]
    return cmds


def build_analysis_commands(cfg: dict) -> list[list[str]]:
    return [
        [_python(), "analysis/generate_paper_figures.py",
         "--results-dir", cfg["output"]["results_dir"]],
    ]

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_commands(
    cmds: list[list[str]],
    phase_name: str,
    dry_run: bool,
):
    print(f"\n{'='*60}")
    print(f"Phase: {phase_name}  ({len(cmds)} command(s))")
    print(f"{'='*60}")

    for i, cmd in enumerate(cmds, 1):
        cmd_str = " ".join(cmd)
        print(f"\n  [{i}/{len(cmds)}] {cmd_str}")
        if dry_run:
            print("         (dry-run, skipped)")
            continue
        try:
            subprocess.run(cmd, cwd=_PROJECT_ROOT, env=_SUBPROCESS_ENV, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"  WARNING: command exited with code {exc.returncode}")
        except FileNotFoundError:
            print(f"  WARNING: command not found")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    config_path = os.path.join(_PROJECT_ROOT, args.config)
    if not os.path.isfile(config_path):
        print(f"ERROR: config not found: {config_path}")
        sys.exit(1)

    cfg = load_config(config_path)
    seeds = args.seeds or cfg["experiment"]["seeds"]
    budget = args.time_budget or cfg["experiment"]["time_budget"]

    print("=" * 60)
    print("KAO Master Runner")
    print("=" * 60)
    print(f"  Config  : {args.config}")
    print(f"  Phase   : {args.phase}")
    print(f"  Seeds   : {seeds}")
    print(f"  Budget  : {budget}s")
    print(f"  Dry-run : {args.dry_run}")

    phase = args.phase

    if phase in ("all", "kao"):
        run_commands(build_kao_commands(cfg, seeds, budget), "KAO", args.dry_run)

    if phase in ("all", "baselines"):
        run_commands(build_baseline_commands(cfg, seeds, budget), "Baselines", args.dry_run)

    if phase in ("all", "ablation"):
        run_commands(build_ablation_commands(cfg, seeds, budget), "Ablation", args.dry_run)

    if phase in ("all", "pareto"):
        run_commands(build_pareto_commands(cfg, seeds, budget), "Pareto", args.dry_run)

    if phase in ("all", "srbench"):
        run_commands(build_srbench_commands(cfg, seeds, budget), "SRBench", args.dry_run)

    if phase in ("all", "eicu"):
        run_commands(build_eicu_commands(cfg, seeds, budget), "eICU", args.dry_run)

    if phase in ("all", "analysis"):
        run_commands(build_analysis_commands(cfg), "Analysis", args.dry_run)

    print("\n" + "=" * 60)
    print("Master runner complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
