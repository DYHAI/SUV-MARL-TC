#!/usr/bin/env python3
"""Replace marked result-table blocks in main.tex with generated tables."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "paper"
MAIN = DOC / "main.tex"
GENERATED = DOC / "generated_primary"

TABLES = {
    "surface_filter": "surface_filter_primary.tex",
    "classical_planners": "classical_planner_comparison_primary.tex",
    "formation_comparison": "surface_formation_primary.tex",
    "underwater_ippo": "underwater_ippo_primary.tex",
    "planner_stratified": "planner_stratified_primary.tex",
}


def replace_block(text: str, name: str, replacement: str) -> str:
    pattern = re.compile(
        rf"% PRIMARY_TABLE_START: {re.escape(name)}.*?"
        rf"% PRIMARY_TABLE_END: {re.escape(name)}",
        flags=re.DOTALL,
    )
    matches = pattern.findall(text)
    if len(matches) != 1:
        raise ValueError(f"{name}: expected one marker block, found {len(matches)}")
    block = (
        f"% PRIMARY_TABLE_START: {name}\n"
        f"{replacement.rstrip()}\n"
        f"% PRIMARY_TABLE_END: {name}"
    )
    return pattern.sub(lambda _match: block, text)


def main() -> None:
    missing = [
        filename
        for filename in TABLES.values()
        if not (GENERATED / filename).is_file()
    ]
    if missing:
        raise SystemExit(f"waiting for generated tables: {sorted(set(missing))}")
    text = MAIN.read_text(encoding="utf-8")
    for name, filename in TABLES.items():
        replacement = (GENERATED / filename).read_text(encoding="utf-8")
        text = replace_block(text, name, replacement)
    MAIN.write_text(text, encoding="utf-8")
    if os.environ.get("SKIP_COMPILE", "0") != "1":
        subprocess.run(
            [
                str(Path.home() / ".local/bin/tectonic"),
                "-X",
                "compile",
                "main.tex",
                "--outdir",
                ".",
                "--keep-logs",
            ],
            cwd=DOC,
            check=True,
        )
    print("updated result tables and compiled main.tex")


if __name__ == "__main__":
    main()
