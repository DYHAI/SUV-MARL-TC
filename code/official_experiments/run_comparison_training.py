#!/usr/bin/env python3
"""Train formation and IPPO/SAC/TD3 comparisons under constrained dynamics."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from code.official_experiments.run_main_training import common_env, process_running
from code.paths import REPO_ROOT, RUNS_ROOT


EXPERIMENT = RUNS_ROOT / "official"
ROOT = REPO_ROOT
PYTHON = sys.executable
MAX_JOBS = int(os.environ.get("DYN_COMPARISON_JOBS", "2"))
STEPS = int(os.environ.get("DYN_TRAIN_STEPS", "2000000"))
COMPARISON_SCOPE = os.environ.get("DYN_COMPARISON_SCOPE", "all")
SURFACE_TAG = os.environ.get("DYN_SURFACE_TAG", "").strip()
SKIP_OFFPOLICY = os.environ.get("DYN_SKIP_OFFPOLICY", "0") == "1"
ALGO_SEEDS = [
    int(value) for value in os.environ.get("DYN_ALGO_SEEDS", "0 1").split()
]


@dataclass(frozen=True)
class Job:
    name: str
    command: list[str]
    env: dict[str, str]
    log: Path
    output: Path


def jobs() -> list[Job]:
    out: list[Job] = []
    formation_suffix = f"_{SURFACE_TAG}" if SURFACE_TAG else ""
    formation_dir = EXPERIMENT / "models" / f"formation{formation_suffix}"
    formation_logs = EXPERIMENT / "logs" / f"formation{formation_suffix}"
    formation_dir.mkdir(parents=True, exist_ok=True)
    formation_logs.mkdir(parents=True, exist_ok=True)
    if COMPARISON_SCOPE != "underwater":
        for speed in (12.0, 20.0):
            for formation in ("line", "grid", "v"):
                name = f"formation_{formation}_n8_v{speed:g}_s0"
                env = common_env()
                env.update(
                    {
                        "FLEET_VMAX": str(speed),
                        "FLEET_MODEL_DIR": str(formation_dir),
                        "FORMATION": formation,
                        "FORMATION_RADIUS_KM": "100",
                        "SEED": "0",
                        "NENV": os.environ.get("DYN_NENV", "4"),
                        "DEPLOY_OCEAN_ONLY": "1",
                        "DEPLOY_MIN_COAST_KM": "100",
                        "SAFETY_MARGIN_KM": "100",
                        "OBS_AGENT_ID": "0",
                        "SST_MODE": "none",
                        "USE_MSLP": "0",
                        "STORAGE16": "0",
                    }
                )
                out.append(
                    Job(
                        name=name,
                        command=[
                            PYTHON,
                            "-m",
                            "code.surface.train_fleet_segment_actfilter",
                            str(STEPS),
                            f"{name}.zip",
                            "0.8",
                            "8",
                            "800",
                        ],
                        env=env,
                        log=formation_logs / f"{name}.log",
                        output=formation_dir / f"{name}.zip",
                    )
                )

    algo_suffix = f"_{SURFACE_TAG}" if SURFACE_TAG and COMPARISON_SCOPE == "surface" else ""
    algo_dir = EXPERIMENT / "models" / f"algorithms{algo_suffix}"
    algo_logs = EXPERIMENT / "logs" / f"algorithms{algo_suffix}"
    algo_dir.mkdir(parents=True, exist_ok=True)
    algo_logs.mkdir(parents=True, exist_ok=True)
    if not SKIP_OFFPOLICY:
        for seed in ALGO_SEEDS:
            tasks = ("underwater",) if COMPARISON_SCOPE == "underwater" else (
                ("surface",) if COMPARISON_SCOPE == "surface" else ("surface", "underwater")
            )
            for task in tasks:
                for algorithm in ("sac", "td3"):
                    if task == "surface":
                        name = f"surface_{algorithm}_n8_v12_s{seed}"
                    else:
                        name = f"underwater_{algorithm}_n4_v1.5_d30_s{seed}"
                    env = common_env()
                    env.update({"ALGO_MODEL_DIR": str(algo_dir)})
                    command = [
                        PYTHON,
                        "-m",
                        "code.algorithm_comparison.train_offpolicy_representative",
                        task,
                        algorithm,
                        "--steps",
                        str(STEPS),
                        "--workers",
                        "4",
                        "--seed",
                        str(seed),
                    ]
                    out.append(
                        Job(
                            name=name,
                            command=command,
                            env=env,
                            log=algo_logs / f"{name}.log",
                            output=algo_dir / f"{name}.zip",
                        )
                    )
    return out


def main() -> None:
    marker = Path(
        os.environ.get(
            "DYN_COMPARISON_MARKER",
            str(EXPERIMENT / "COMPARISON_TRAINING.done"),
        )
    )
    state_path = Path(
        os.environ.get(
            "DYN_COMPARISON_STATE",
            str(EXPERIMENT / "comparison_training_state.json"),
        )
    )
    state_path.parent.mkdir(parents=True, exist_ok=True)
    lock_stream = (state_path.parent / f"{state_path.name}.lock").open("a+")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(
            f"[comparison] another scheduler owns {lock_stream.name}; exiting",
            flush=True,
        )
        return
    lock_stream.seek(0)
    lock_stream.truncate()
    lock_stream.write(f"pid={os.getpid()} updated={time.strftime('%F %T')}\n")
    lock_stream.flush()
    all_jobs = jobs()
    pending = [job for job in all_jobs if not job.output.exists()]
    running: dict[str, tuple[Job, subprocess.Popen, float, object]] = {}
    failures: list[str] = []
    print(
        f"[comparison] total={len(all_jobs)} complete={len(all_jobs)-len(pending)} "
        f"pending={len(pending)} max_jobs={MAX_JOBS}",
        flush=True,
    )
    while pending or running:
        pending = [job for job in pending if not job.output.exists()]
        external = {
            job.name
            for job in pending
            if job.name not in running
            and process_running(job.name, Path(job.command[1]).name)
        }
        available = MAX_JOBS - len(running) - len(external)
        while available > 0:
            candidate = next(
                (
                    job
                    for job in pending
                    if job.name not in running and job.name not in external
                ),
                None,
            )
            if candidate is None:
                break
            pending.remove(candidate)
            candidate.log.parent.mkdir(parents=True, exist_ok=True)
            stream = candidate.log.open("a", buffering=1)
            stream.write(f"\n[{time.strftime('%F %T')}] launch from {ROOT}\n")
            proc = subprocess.Popen(
                candidate.command,
                cwd=ROOT,
                env=candidate.env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            running[candidate.name] = (
                candidate,
                proc,
                time.time(),
                stream,
            )
            print(f"[start] {candidate.name} pid={proc.pid}", flush=True)
            available -= 1
        time.sleep(20)
        for name, (job, proc, started, stream) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            stream.close()
            del running[name]
            elapsed = (time.time() - started) / 60.0
            if rc == 0 and job.output.exists():
                print(f"[done] {job.name} {elapsed:.1f} min", flush=True)
            else:
                failures.append(name)
                print(
                    f"[fail] {job.name} rc={rc} elapsed={elapsed:.1f} min "
                    f"log={job.log}",
                    flush=True,
                )
        state = {
            "updated": time.strftime("%F %T"),
            "running": sorted(running),
            "external_running": sorted(external),
            "completed": len([j for j in all_jobs if j.output.exists()]),
            "total": len(all_jobs),
            "pending": len(pending),
            "failures": failures,
        }
        state_path.write_text(json.dumps(state, indent=2) + "\n")
    if failures:
        raise SystemExit("failed jobs: " + ", ".join(failures))
    marker.touch()
    print("[comparison] complete", flush=True)


if __name__ == "__main__":
    main()
