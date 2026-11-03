"""Shared repository paths for the paper-facing code."""

from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CODE_ROOT.parent
DATA_ROOT = REPO_ROOT / "data"
PAPER_ROOT = REPO_ROOT / "paper"
RESULTS_ROOT = REPO_ROOT / "results"
RUNS_ROOT = REPO_ROOT / "runs"
