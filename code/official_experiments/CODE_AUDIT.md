# Code audit for the official constrained-dynamics version

Audit date: 2026-11-06; re-audited 2026-11-07 and 2026-11-09.

## Scope

The audit covers the implementation and runners used by the official
manuscript experiments:

- `constrained_dynamics.py`
- `fleet_segment_env.py`
- `underwater_fleet_flow_env.py`
- final surface and subsurface training wrappers
- learned-policy and traditional-planner evaluation scripts
- all resumable experiment schedulers in this directory

The constrained-dynamics models are now the official manuscript models. The
final verification report is stored in `verification_report.json`.

## Bugs found and fixed

1. SAC/TD3 evaluation attempted to load every `.zip` model as PPO, causing
   `use_sde` keyword collisions. The evaluators now select PPO, SAC, or TD3
   from the model filename.
2. Underwater classical evaluation omitted the position reached by the final
   action, shortening reported paths and coverage by one control step.
3. The classical underwater duration denominator used one more time step than
   the recorded path. Path, ground speed, and duration are now mutually
   consistent.
4. Learned underwater evaluation also omitted the final state from path and
   coverage. The final state and sampled current are now included.
5. Surface HYCOM 0 m interpolation rebuilt two SciPy interpolators every
   control step. It now uses direct bilinear grid interpolation.
6. Surface 0 m current was coupled to constrained dynamics. Current advection
   and acceleration/heading constraints are now independent switches, allowing
   both current-only and current-plus-constraints experiments.
7. Constrained observations now include `cos(heading)` and `sin(heading)`.
   The current-only surface variant receives the same two redundant values so
   the two new surface variants have identical observation dimensions.
8. `UnderwaterFleetFlowEnv.reset(seed=...)` accepted a seed but still sampled
   every episode from NumPy's global RNG. Subprocess workers therefore began
   from identical states and could reuse the same TC segment/deployment. The
   explicit seed is now applied before episode and deployment sampling.
   This bug predates the isolated constrained-dynamics study and therefore
   also affects reproducibility of the frozen underwater models generated
   before 2026-11-07. Those manuscript artifacts were deliberately left
   untouched; strict regeneration would require retraining the historical
   underwater matrix rather than only re-running its evaluation.
9. The underwater evaluation driver conflated the number of trained models
   with the number of paired deployment seeds. A one-training-seed run could
   therefore only evaluate one deployment per TC. These quantities are now
   independent, and the constrained main matrix evaluates ten deployments per
   held-out TC.
10. Classical underwater evaluation appended the final position but not the
    final sampled current speed, while learned-policy evaluation included both.
    The final current sample is now included in both paths.
11. The surface action-filter evaluator defaulted to the historical 64-TC
    split and reused one deployment bearing across all test segments. The
    constrained experiment now exports the 500-TC split and uses a
    segment-dependent seed with the same ocean-aware deployment sampler.
12. The optional 3x3 frame-stack observation path referenced undefined local
    interpolators when `N > 1`. It now builds the current frame explicitly.
13. Surface verification expected 16 action-filter runs although the scheduler
    creates 8, and did not verify the episode count. It now checks the actual
    8-run matrix and 768 held-out episodes per run.
14. Final verification required `ALL_CONSTRAINED_EXPERIMENTS.done`, but that
    marker is created only after verification succeeds, creating a logical
    deadlock. The marker is now treated as an output of successful
    verification rather than an input to it.
15. The same Gymnasium seed/global-NumPy mismatch existed in seven older
    environment classes used by historical scripts. Explicit reset seeds now
    also seed the global NumPy stream used by those classes, while unseeded
    calls keep their previous behavior.
16. The surface evaluator estimated maximum propulsion speed from the norm of
    the projected action, which could print values such as `12.000002 m/s`
    because of floating-point roundoff. The reported value is now bounded by
    the configured nominal maximum, and final verification uses a small
    numerical tolerance.
17. Surface evaluation originally reloaded the same 50-storm current archive
    for every fleet/radius task. A batch evaluator now reuses one preload across
    the five radii while retaining the existing rollout function. A 50-segment
    regression produced byte-identical `ROW`, `FINALROW`, and `CONSTRAINTROW`
    values to the single-task evaluator.
18. `train_uwflow_sb3.py` constructed forked workers without an explicit reset
    seed. The environment supported `reset(seed=...)`, but every worker still
    inherited the same global NumPy state and could replay the same episode and
    deployment stream. Each worker is now initialized with `SEED + worker_id`
    before SB3's first unseeded reset.
19. The constrained surface classical-planner evaluator hard-coded a 0.2
    low-pass coefficient, while IPPO and the SAC/TD3 representatives use 0.8.
    The coefficient is now configurable, the constrained matrix explicitly
    exports 0.8, and each classical result records the value used.
20. The surface action-filter scheduler ignored existing `.done` markers and
    reran completed jobs. Both the active variant scheduler and the older
    post-training runner now skip completed action-filter runs.
21. Classical planner CSVs were written directly to their final paths, so a
    killed process could leave a non-empty partial file that the scheduler
    treated as complete. Both classical evaluators now assert the exact row
    count and publish through an atomic temporary file; schedulers also skip
    only files with the complete expected row count.
22. Underwater evaluation defaulted to four simultaneous 50-TC field preloads.
    With the surface matrices resident in memory, this could exhaust RAM. The
    default is reduced to two; the panel parallelism remains configurable.
23. Multiple copies of a resumable scheduler could launch the same model before
    either saw the other's child process. Main and comparison schedulers now
    take non-blocking per-state-file locks, and PPO/SAC/TD3 saves publish through
    atomic temporary files.
24. Stable-Baselines3 2.9 writes exactly to the supplied temporary path rather
    than appending `.zip`. The first atomic-save implementation therefore
    trained `n16/v20` but failed during the final rename. The temporary model
    was retained and promoted after validation; save/replace names are now
    identical.
25. Two surface batch evaluators started the same `n16/v20` rollout before
    either observed the other's `.done` marker. Batch evaluation now takes a
    non-blocking lock per speed and fleet size, in addition to skipping
    completed radii.
26. Final verification now loads every expected PPO/SAC/TD3 artifact and
    checks that it completed at least two million training steps, in addition
    to validating evaluation tables and completion markers.
27. Individual `eval_deploy_xfer.py` runs now take a deterministic lock keyed by
    model, fleet size, deployment radius, speed, split, action filter, and
    formation. This prevents the official evaluator and a resumed/pre-evaluation
    shell from rolling out the same configuration concurrently. A duplicate
    invocation exits with temporary-failure status so its caller never creates
    a false `.done` marker.
28. `DYN_SKIP_OFFPOLICY=1` now removes SAC/TD3 training and evaluation from the
    resumable comparison pipeline, allowing the primary one-seed matrices to run
    first as requested.
29. External-process detection matched a model filename anywhere in a Python
    command, so a formation evaluation of `surface_n8_v20_s0.zip` blocked the
    same-named training job. Detection now also requires the job's `train_*`
    script name in the command line.
30. Added a dedicated `dynamics_only` surface variant. It uses the paper's
    sea-surface ground-speed convention, does not add HYCOM current to motion
    or observations, and retains only the constrained acceleration, heading,
    yaw-rate, and minimum-turn-radius dynamics. The earlier current-only and
    current-plus-constrained experiments remain isolated sensitivity runs.

The completed surface SAC artifact was moved, not deleted, to
`backup_surface_offpolicy_skipped_20261107/`. Final surface verification now
accepts an empty algorithm-comparison directory when that comparison is
explicitly skipped.

Artifacts produced before fixes 18 and 19 were moved, not deleted:

- `backup_underwater_worker_seed_20261107/` contains the 14 underwater IPPO
  models, their main/algorithm evaluations, completion markers, and scheduler
  state generated before per-worker seeding was fixed.
- `backup_surface_classical_beta02_20261107/` contains the 12 completed surface
  classical-planner CSVs and their logs generated with the mismatched 0.2
  low-pass coefficient.

## Verified invariants

- Original kinematic mode still uses surface observation dimension 14.
- Constrained mode uses two additional heading-observation values.
- Surface and subsurface propulsion speed never exceeds their configured
  `Vmax`.
- Longitudinal acceleration and deceleration stay within their configured
  limits.
- Yaw rate stays within its configured limit and the minimum turn radius
  constraint when the vehicle is moving.
- HYCOM 0 m current affects only the surface motion equation, not the actor
  observation.
- Identical seeds and actions reproduce identical observations/positions.
- With zero surface current and original mode disabled, the pre-existing
  behavior is preserved.

## Validation executed

- `python -m compileall -q .`
- `bash -n` over all project shell scripts
- `python tests/run_tests.py`: 7/7 passed
- `validate_dynamics.py`: all speed/acceleration/yaw assertions passed
- `pyflakes` over all 317 Python files: no syntax or undefined-name errors
- randomized 500-step dynamics checks: speed, acceleration, yaw-rate, current
  advection, and finite-output invariants passed for surface and subsurface
- seeded-reset checks: identical seeds reproduce observations and deployments;
  different seeds produce different deployments
- worker-initialization check: underwater vector-env workers receive
  `SEED + worker_id`
- current-only surface training smoke test: passed
- current-plus-constraints surface training smoke test: passed
- constrained subsurface training smoke test: passed
- PPO/SAC/TD3 learned-policy loading and evaluation: passed
- 56/56 classical subsurface configurations, 500 episodes each:
  passed after the final-state fix

## Remaining modeling limitations

- The acceleration and yaw limits are conservative modeling assumptions, not
  calibrated Blue Whale measurements.
- HYCOM 0 m velocity excludes Stokes drift and wind-driven surface drift.
- Suspended processes preserve in-memory progress but are not checkpointed
  against a machine reboot.
