#!/usr/bin/env python3
"""Resumable scheduler for the constrained-dynamics main training matrix.

Surface: 4/8/12/16 SUVs x 12/20 m/s x 10 seeds = 80 models.
Subsurface: 7 depths x 1.5/2.0 m/s x 10 seeds = 140 models.

All artifacts stay under ``constrained_dynamics_experiments``.  Existing
manuscript models and result files are never read or overwritten.
"""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from code.paths import REPO_ROOT, RUNS_ROOT

EXPERIMENT = RUNS_ROOT / "official"
ROOT = REPO_ROOT
PYTHON = sys.executable
MAX_JOBS = int(os.environ.get("DYN_MAX_JOBS", "6"))
STEPS = int(os.environ.get("DYN_TRAIN_STEPS", "2000000"))
NENV = int(os.environ.get("DYN_NENV", "5"))
SURFACE_ENABLED = os.environ.get("DYN_SURFACE_ENABLED", "1") == "1"
UNDERWATER_ENABLED = os.environ.get("DYN_UNDERWATER_ENABLED", "1") == "1"
UNDERWATER_JOBS = int(os.environ.get("DYN_UNDERWATER_JOBS", "7"))
SURFACE_JOBS = int(os.environ.get("DYN_SURFACE_JOBS", "2"))
SEED_LIST = [
    int(value) for value in os.environ.get("DYN_SEEDS", "0 1 2 3 4 5 6 7 8 9").split()
]


@dataclass(frozen=True)
class Job:
    name: str
    kind: str
    command: list[str]
    env: dict[str, str]
    log: Path
    output: Path


def common_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "DYNAMICS_MODE": env.get("DYNAMICS_MODE", "constrained"),
            "DYN_SURFACE_CURRENT": env.get("DYN_SURFACE_CURRENT", "0"),
            "DYN_SURFACE_ACC_MS2": env.get("DYN_SURFACE_ACC_MS2", "0.02"),
            "DYN_SURFACE_DEC_MS2": env.get("DYN_SURFACE_DEC_MS2", "0.03"),
            "DYN_SURFACE_YAW_DEG_S": env.get("DYN_SURFACE_YAW_DEG_S", "1.0"),
            "DYN_SURFACE_MIN_TURN_RADIUS_M": env.get(
                "DYN_SURFACE_MIN_TURN_RADIUS_M", "200"
            ),
            "DYN_SUB_ACC_MS2": env.get("DYN_SUB_ACC_MS2", "0.002"),
            "DYN_SUB_DEC_MS2": env.get("DYN_SUB_DEC_MS2", "0.003"),
            "DYN_SUB_YAW_DEG_S": env.get("DYN_SUB_YAW_DEG_S", "2.0"),
            "DYN_SUB_MIN_TURN_RADIUS_M": env.get(
                "DYN_SUB_MIN_TURN_RADIUS_M", "100"
            ),
            "SPLIT_JSON": str(ROOT / "data" / "typhoon_split_500.json"),
            "SEGMENTS_CSV": str(ROOT / "data" / "segments_500.csv"),
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "CUDA_VISIBLE_DEVICES": "",
        }
    )
    return env


def surface_jobs() -> list[Job]:
    jobs = []
    model_subdir = os.environ.get("DYN_SURFACE_MODEL_SUBDIR", "surface_main")
    log_subdir = os.environ.get("DYN_SURFACE_LOG_SUBDIR", "surface_main")
    model_dir = EXPERIMENT / "models" / model_subdir
    log_dir = EXPERIMENT / "logs" / log_subdir
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    for seed in SEED_LIST:
        for speed in (12.0, 20.0):
            for n_auv in (4, 8, 12, 16):
                name = f"surface_n{n_auv}_v{speed:g}_s{seed}"
                env = common_env()
                env.update(
                    {
                        "FLEET_VMAX": str(speed),
                        "FLEET_MODEL_DIR": str(model_dir),
                        "SEED": str(seed),
                        "NENV": str(NENV),
                        "DEPLOY_OCEAN_ONLY": "1",
                        "DEPLOY_MIN_COAST_KM": "100",
                        "SAFETY_MARGIN_KM": "100",
                        "OBS_AGENT_ID": "0",
                        "SST_MODE": "none",
                        "USE_MSLP": "0",
                        "STORAGE16": "0",
                    }
                )
                jobs.append(
                    Job(
                        name=name,
                        kind="surface",
                        command=[
                            PYTHON,
                            "-m",
                            "code.surface.train_fleet_segment_actfilter",
                            str(STEPS),
                            f"{name}.zip",
                            "0.8",
                            str(n_auv),
                            "800",
                        ],
                        env=env,
                        log=log_dir / f"{name}.log",
                        output=model_dir / f"{name}.zip",
                    )
                )
    return jobs


def underwater_jobs() -> list[Job]:
    jobs = []
    model_dir = EXPERIMENT / "models" / "underwater_main"
    log_dir = EXPERIMENT / "logs" / "underwater_main"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    for seed in SEED_LIST:
        for depth in (0, 10, 20, 30, 40, 50, 60):
            for speed in (1.5, 2.0):
                name = f"uw_d{depth}_v{speed:g}_s{seed}"
                env = common_env()
                env.update(
                    {
                        "UWFLOW_MODEL_DIR": str(model_dir),
                        "PRELOAD": "1",
                    }
                )
                jobs.append(
                    Job(
                        name=name,
                        kind="underwater",
                        command=[
                            PYTHON,
                            "-m",
                            "code.underwater.train_uwflow_sb3",
                            str(STEPS),
                            name,
                            str(speed),
                            "C",
                            str(depth),
                            "20",
                            str(seed),
                            "0.25",
                            "200",
                            str(NENV),
                        ],
                        env=env,
                        log=log_dir / f"{name}.log",
                        output=model_dir / f"{name}.zip",
                    )
                )
    return jobs


def interleave(*groups: list[Job]) -> list[Job]:
    out: list[Job] = []
    offsets = [0] * len(groups)
    while True:
        added = False
        for k, group in enumerate(groups):
            if offsets[k] < len(group):
                out.append(group[offsets[k]])
                offsets[k] += 1
                added = True
        if not added:
            return out


def process_running(name: str, script_name: str | None = None) -> bool:
    needle = name.encode()
    script_needle = script_name.encode() if script_name else None
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            cmdline = (proc / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if (
            needle in cmdline
            and b"python" in cmdline
            and (script_needle is None or script_needle in cmdline)
        ):
            return True
    return False


def run() -> None:
    state_path = Path(
        os.environ.get(
            "DYN_STATE_FILE", str(EXPERIMENT / "main_training_state.json")
        )
    )
    state_path.parent.mkdir(parents=True, exist_ok=True)
    lock_stream = (state_path.parent / f"{state_path.name}.lock").open("a+")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(
            f"[constrained] another scheduler owns {lock_stream.name}; exiting",
            flush=True,
        )
        return
    lock_stream.seek(0)
    lock_stream.truncate()
    lock_stream.write(f"pid={os.getpid()} updated={time.strftime('%F %T')}\n")
    lock_stream.flush()
    marker = Path(
        os.environ.get(
            "DYN_COMPLETE_MARKER", str(EXPERIMENT / "MAIN_TRAINING.done")
        )
    )
    if SURFACE_ENABLED and UNDERWATER_ENABLED:
        jobs = interleave(surface_jobs(), underwater_jobs())
    elif SURFACE_ENABLED:
        jobs = surface_jobs()
    elif UNDERWATER_ENABLED:
        jobs = underwater_jobs()
    else:
        raise ValueError("both surface and underwater jobs are disabled")
    pending = [job for job in jobs if not job.output.exists()]
    print(
        f"[constrained] total={len(jobs)} complete={len(jobs)-len(pending)} "
        f"pending={len(pending)} max_jobs={MAX_JOBS}",
        flush=True,
    )
    running: dict[str, tuple[Job, subprocess.Popen, float, object]] = {}
    failures: list[str] = []
    while pending or running:
        pending = [job for job in pending if not job.output.exists()]
        external = {
            job.name
            for job in pending
            if job.name not in running
            and process_running(job.name, Path(job.command[1]).name)
        }
        active_jobs = [
            running[name][0] for name in running
        ] + [job for job in pending if job.name in external]
        surface_pending = any(job.kind == "surface" for job in pending)
        limits = {
            "surface": SURFACE_JOBS if surface_pending else 0,
            "underwater": 1 if surface_pending else UNDERWATER_JOBS,
        }
        started_this_pass = True
        while started_this_pass:
            started_this_pass = False
            for kind in ("surface", "underwater"):
                active_kind = sum(job.kind == kind for job in active_jobs)
                if active_kind >= limits[kind]:
                    continue
                candidate = next(
                    (
                        job
                        for job in pending
                        if job.kind == kind
                        and job.name not in running
                        and job.name not in external
                    ),
                    None,
                )
                if candidate is None:
                    continue
                pending.remove(candidate)
                job = candidate
                job.log.parent.mkdir(parents=True, exist_ok=True)
                stream = job.log.open("a", buffering=1)
                stream.write(
                    f"\n[{time.strftime('%F %T')}] launch from {ROOT}\n"
                )
                proc = subprocess.Popen(
                    job.command,
                    cwd=ROOT,
                    env=job.env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                running[job.name] = (job, proc, time.time(), stream)
                active_jobs.append(job)
                print(f"[start] {job.name} pid={proc.pid}", flush=True)
                started_this_pass = True
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
            "completed": len([j for j in jobs if j.output.exists()]),
            "total": len(jobs),
            "pending": len(pending),
            "failures": failures,
        }
        state_path.write_text(json.dumps(state, indent=2) + "\n")
    if failures:
        raise SystemExit("failed jobs: " + ", ".join(failures))
    marker.touch()
    print("[constrained] main training complete", flush=True)


if __name__ == "__main__":
    run()
