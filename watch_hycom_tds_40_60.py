#!/usr/bin/env python3
"""Keep the tds.hycom.org 40/60 m download pool alive until both depths are
complete. Unlike watch_hycom_remaining.py, this supervisor always uses the
tds.hycom.org NCSS host and larger chunks (48 h for the pre-2019 chain,
24 h for the 2019+ chain). It only launches a family when the active count
for that family is below its target, so it can run alongside the manual
tds workers already launched."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import audit_hycom_currents


ROOT = Path(__file__).resolve().parent
RL = "/home/ding/main/miniconda3/envs/RL/bin/python"
NAS = "/mnt/nas/multi_auv_typhoon_tracking/union_current_depths"
LOGDIR = ROOT / "logs_hycom_depth_10_20_40_60"
OLD_WORKERS = int(os.environ.get("HYCOM_TDS_OLD_WORKERS", "4"))
NEW_WORKERS = int(os.environ.get("HYCOM_TDS_NEW_WORKERS", "2"))
NCSS_BASE = "https://tds.hycom.org/thredds/ncss/grid/"


def running_commands() -> list[str]:
    commands: list[str] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if raw:
            commands.append(raw.replace(b"\0", b" ").decode(errors="replace"))
    return commands


def active(depth: int, family: str) -> int:
    if family == "old":
        needle = "download_hycom_500.py dyn "
        suffix = f" uv{depth}"
    else:
        needle = "download_hycom_200.py dyn "
        suffix = f" {depth}"
    python_needle = f"/envs/RL/bin/python {needle}"
    return sum(
        python_needle in cmd and cmd.rstrip().endswith(suffix)
        for cmd in running_commands()
    )


def launch(depth: int, family: str, worker: int) -> None:
    LOGDIR.mkdir(parents=True, exist_ok=True)
    if family == "old":
        script = ["download_hycom_500.py", "dyn", str(worker), f"uv{depth}"]
        env_update = {"CHUNK_H": "48", "MIN_CHUNK_H": "3"}
        log = LOGDIR / f"tds_super_d{depth}_old_w{worker}.log"
    else:
        script = ["download_hycom_200.py", "dyn", str(worker), str(depth)]
        env_update = {"CHUNK_H": "24", "MIN_CHUNK_H": "3"}
        log = LOGDIR / f"tds_super_d{depth}_new_w{worker}.log"
    env = os.environ.copy()
    env.update(env_update)
    env.update(
        {
            "NCSS_BASE": NCSS_BASE,
            "HYCOM_DOWNLOAD_ROOT": NAS,
            "HYCOM_TIGHT_BBOX": "1",
            "AREA_MODE": "tight",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    with log.open("ab") as handle:
        subprocess.Popen(
            [RL, *script],
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def main() -> None:
    while True:
        report = audit_hycom_currents.audit((40, 60))
        pending = []
        for depth in (40, 60):
            available = int(report["depths"][str(depth)]["available"])
            print(
                f"{time.strftime('%F %T')} tds depth={depth} m "
                f"available={available}/500",
                flush=True,
            )
            if available == 500:
                continue
            pending.append(depth)
            old_count = active(depth, "old")
            new_count = active(depth, "new")
            for worker in range(old_count + 1, OLD_WORKERS + 1):
                launch(depth, "old", worker)
            for worker in range(new_count + 1, NEW_WORKERS + 1):
                launch(depth, "new", worker)
        if not pending:
            print(f"{time.strftime('%F %T')} tds 40/60 m archives complete", flush=True)
            return
        time.sleep(300)


if __name__ == "__main__":
    main()
