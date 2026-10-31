# Revised Manuscript: IPPO Terminology Preserved

Files:

- `main_revised_keep_ippo.tex`
- `main_revised_keep_ippo.pdf`
- `MARL_Ding.bib`

## Main corrections

1. The abstract now lists the final simulated depths explicitly as 0, 10, 20,
   30, 40, 50, and 60 m.
2. The former conceptual weighted objective \(J\) and the unimplemented hard
   energy/coast/communication constraints were replaced by an accurate
   description of the observation objectives and operational conditions.
3. The surface reward now matches the implementation:
   - unbounded indicator placement error corrected;
   - near-center bonus represented as an additive indicator term;
   - \(R_{\mathrm{form}}\) and \(W_{\mathrm{form}}\) aligned with the code.
4. The underwater reward now uses the implemented normalized-current,
   propulsion-energy, and downstream-alignment terms. The former unified
   \(\lambda_v,\lambda_c,\lambda_u\) formula was removed because it did not
   describe a single implemented reward mode.
5. IPPO terminology is retained, but the description now states that:
   - actor and critic weights are shared;
   - actions are sampled independently for each SUV;
   - the implementation uses a team-average cooperative reward;
   - the shared critic is evaluated on each local observation and the values
     are summed;
   - no global-state centralized critic is used.
6. Observation assumptions were corrected:
   - the final wind-only surface model is memoryless;
   - ideal sharing of all SUVs' sampled wind observations is assumed in the
     main surface experiments;
   - the active underwater IPPO comparison uses a five-frame local current history;
   - the recommended underwater mode is passive drift.
7. Coastal safety now includes the actual 100 km deployment threshold,
   100 km soft buffer, and 100 km action-filter safety margin.
8. The data table explicitly reports 499/500 completeness for the 0 m HYCOM
   product instead of claiming all products cover all 500 events.
9. The final experiment protocol now specifies:
   - surface wind-only IPPO for 4/8/12/16 SUVs with fixed `360/N` angles;
   - random training deployment inside an 800 km disk;
   - fixed-radius validation at 100/200/400/600/800 km;
   - active IPPO, random control, and passive drift at seven depths;
   - ten seeds and the 100 km coastal-safety settings.
10. Qualitative Results and Conclusion sections were updated for this final
    protocol. Numerical values still need to be inserted after all final runs
    are frozen.
11. Appendix tables were reduced to two merged tables and placed before the
    bibliography. The bibliography is generated after the appendices.
12. The 800 km-disk training and 100/200/400/600/800 km fixed-radius validation
    protocol is described in the Results text.
13. A conditional underwater energy/path figure block was added. The plotting
    script is `../plot_final_underwater_results.py`; it will produce the figure
    only after all seven depths, both speed limits, and all three methods are
    available.
14. Figure 2 (the upper-ocean response schematic) was removed from the
    manuscript for now, as requested.
15. The surface and underwater reward equations were split over additional
    lines to prevent inline symbols from overlapping adjacent body text.
16. `\refstepcounter{table}` was removed from the appendix tables because
    `\captionof{table}` already increments the counter. The appendix tables are
    now numbered exactly A.1 and A.2.
17. Surface and subsurface vertical coordinates are explicitly distinguished:
    the surface policy observes wind at 10 m above the sea surface, whereas the
    subsurface policy observes ocean currents at 0/10/20/30/40/50/60 m below
    the sea surface.
18. Table 2 columns were renamed to `Public reference`, `High-speed scenario`,
    and `Current-speed scenario`.
19. The former final transfer-test table was removed, so the main manuscript
    now contains only Table 1 (data), Table 2 (platform scenarios), and the two
    appendix tables A.1/A.2.
20. The 800 km training radius is now defined as an explicit upper-bound
    assumption for operational TC-track forecast error. The 100/200/400/600/
    800 km validation radii are described as a controlled mild-to-severe
    displacement series evaluated without retraining.
21. The 450/50 split is now explicitly interpreted as a generalization test on
    unseen TCs: none of the 50 held-out TCs contributes any segment or
    environmental field to training.
22. Manuscript prose uses "tropical cyclone" or "TC" rather than "typhoon".
    Bibliographic titles and external file names retain their original wording.
23. The deployment-error rationale and the 450/50 generalization interpretation
    were also synchronized to the editor's `main.tex`; both `main.tex` and
    `main_revised_keep_ippo.tex` compile successfully with Tectonic.
24. Table 2 was shortened and relabeled with concise
    `Public reference / High speed / Current speed` scenarios.
25. The surface observation text now states explicitly that each SUV shares
    the 10 m wind observation acquired at its own position, so the policy sees
    the fleet-level union of local samples rather than the full ERA5 field.
26. ERA5, HYCOM, and IBTrACS source portals are included as line-broken
    footnotes and cited directly from the Data section.
27. The paired surface/subsurface figure was rebuilt with equal, larger panel
    labels; a dedicated colorbar row removes label overlap; the plotting script
    is `make_fig2_surface_subsurface_observation.py`.
28. The final surface comparison will use ten matched seeds for 12 and
    20 m/s. Tables and a 4-by-4 error-bar figure report seed mean and sample
    standard deviation; `aggregate_surface_vmax_comparison.py` is the single
    generation path.
29. The working manuscript directory is now `Latex_Yihui_MARL/`, containing
    the LaTeX source, rendered PDF, figures, tables, BibTeX files, and figure
    generation script.
30. The fleet definition is written as an inline sentence rather than a
    standalone one-line equation; the 111 km degree-to-distance constant in
    the segmentation equation is explicitly explained.
31. Figure 1 uses a transparent, cropped Blue Whale-class vessel image at all
    four SUV positions, with one common heading. The right-hand title now
    identifies HYCOM as `exampled by 30 m`.
32. Table 2 uses concise `High speed scenario` / `Current speed scenario`
    headings and `0--60 m (7-layer)`.
33. The code-availability section retains the project repository and reports
    runnable software versions plus the workstation CPU, memory, and GPU; the
    dependency-repository URLs were removed.
34. Training uses center distance only in the reward. Validation removes both
    the reward and the center coordinate and tests the frozen policy on unseen
    TCs using shared wind observations.
35. Final paired low-pass and on-land-reward ablations were completed and
    incorporated.
36. The final low-pass table now covers 4/8/12/16 SUVs on the 500-TC
    benchmark over 768 held-out segments. Panel B was removed. The on-land
    reward comparison is instead shown in a separate two-bar figure
    (`6.60%` without reward versus `5.94%` with `P_land=20`).
37. Figure 1 was moved into the Introduction, preceded by a paragraph
    explaining the surface-transit and subsurface-sampling modes. The caption
    states that four SUVs are illustrative and that experiments also cover
    8/12/16 vehicles.
38. The Abstract now describes subsurface observations as the `0--60 m`
    range rather than listing every depth inline.
39. The method explicitly states that SAC/TD3 were not used because their
    replay buffers scale poorly with joint multi-agent observations, actions,
    and parallel training; the final study uses on-policy parameter-sharing
    IPPO.
40. The complete 400-cell ten-seed comparison is now included. The 20 m/s
    policy has lower tracking distance in all 20 fleet-size/radius
    comparisons, with 19 significant at `p<0.05`; the averaged 12-minus-20 m/s
    gap grows from 10.1 km at 100 km to 84.2 km at 800 km. The 12 m/s policy
    uses shorter paths and lower mean speed, so the paper reports this as an
    efficiency--tracking trade-off.
41. Final tables:
    - Table 3: tracking distance and path length;
    - Table 4: mean speed and on-land fraction;
    - Table 5: low-pass ablation for 4/8/12/16 SUVs.
    On-land fraction in Table 4 is reported after multiplication by 100.
42. Final figures:
    - Figure 2: 4-by-4 ten-seed speed/deployment comparison;
    - Figure 3: separate on-land-reward ablation;
    - Figure 4: ten-seed underwater strategy comparison.
43. All 400 seed-level 12/20 m/s evaluations were parsed successfully. The
    final PDF compiles with Tectonic without unresolved references or missing
    inputs.

## Remaining work

- Insert final quantitative Results tables for the completed 10-seed runs.
- Update the qualitative Results text with final means, standard deviations,
  confidence intervals, and statistical test results.
- Verify the final author list, affiliations, CRediT statement, funding, and
  repository/DOI availability statement before JOES submission.

## 2026-10-29 scientific-language and reporting audit

- Replaced `simulation-validated` and equivalent claims with
  `simulation-based`; the manuscript now explicitly states that no field
  validation or operational forecast impact is claimed.
- Defined the reported IPPO implementation precisely: shared actor and critic,
  independent per-vehicle Gaussian action sampling, a joint action
  log-probability in the PPO ratio, a team-average reward, per-agent value
  evaluation, and no global-state centralized critic.
- Renamed the surface metric consistently as `time-averaged distance` and
  reserved `final distance` for the end-of-episode metric. Existing numerical
  claims are now attached only to the time-averaged metric.
- Documented the statistical unit and test: 768 held-out segments are averaged
  within each seed, mean and sample standard deviation are computed over ten
  seeds, and two-sided paired t-tests are applied at the seed level. The
  manuscript states that the twenty comparisons are uncorrected for
  multiplicity and are exploratory rather than confirmatory.
- Replaced `mean observed/measured current speed` with
  `mean experienced current speed`, matching Eq. (16).
- Reduced the abstract to 228 words.
- Softened claims based on representative trajectories and literature,
  including `proved competitive`, `demonstrates capability`, and an
  unsupported data-assimilation benefit claim.
- Clarified that the underwater four-vehicle design tests decision strategies
  and does not estimate fleet-size scaling, because vehicles do not share
  observations underwater.
- Added explicit limitations for reanalysis-derived forcing, horizontal
  point-mass dynamics, absent six-degree-of-freedom effects, no field trials,
  and no downstream data-assimilation assessment.
- Tectonic compilation succeeds with 20 pages and no undefined citations or
  references. The only overfull box above 100 pt originates from the CAS
  title/maketitle block and is not visibly clipped in the rendered first page.
- Not submission-ready until the final-distance surface evaluation and the
  complete seven-depth underwater matrix replace their current
  interim/preview fallbacks.

## 2026-10-29 cooperative-task, Table 3, and large-offset figure revisions

- The IPPO method section now identifies the problem as a fully cooperative
  task: all SUVs optimize the same observation and tracking objective under a
  team-average reward. The manuscript states explicitly that it is neither a
  competitive task nor a mixed cooperative--competitive task and that this
  structure motivates parameter-sharing IPPO.
- Table 3 now includes the fleet-mean final SUV-to-TC-center distance in
  addition to path length, mean speed, and maximum speed. The energy-proxy
  column was removed at the author's request.
- The Table 3 protocol was reproduced exactly with the original benchmark
  settings: 800 km randomized deployment, offshore-only initialization with a
  100 km coast margin, and a 100 km execution-layer coastal safety filter.
  The original path and speed values were recovered before adding final
  distance and energy.
- Final-distance results show that the low-pass filter improves tracking on
  average: the fleet mean decreases from 346.4 to 292.6 km (53.8 km, 15.5%).
  The 4-, 12-, and 16-SUV cases improve, while the 8-SUV case increases only
  slightly by 3.4 km. Across the four fleets, path length, mean speed, and
  maximum speed decrease consistently.
- Figure 4 discussion now separates the two coupled mechanisms at large
  deployment offset: geometric/time-to-intercept limitations and weakening,
  less distinctive remote wind-speed/gradient information. The 600 km example
  still recovers an inward direction, whereas the 800 km example may slow,
  oscillate in direction, or lose the TC.
- The Figure 2 interim caption and its existing Results discussion were left
  unchanged, as requested.
- The embedded title of the 30 m subsurface strategy figure now reads
  `Representative four-SUV subsurface strategies...`.

## 2026-10-29 full consistency audit and implementation corrections

- Abstract, Introduction, and Conclusion now use the same task definition:
  surface and subsurface stages are separate tasks connected by an operational
  handoff; they are not modeled as one continuous mode-switching episode.
- The Abstract is 221 words and reports only completed surface conclusions.
  It no longer claims completed underwater performance before the corrected
  seven-depth matrix is available.
- The Conclusion explicitly defers final underwater conclusions until the
  corrected unit-disk action and causal five-frame history implementation has
  been retrained and evaluated across all seven depths.
- The Dec-POMDP tuple now uses a shared team reward
  $R=(1/N)\sum_i r_i$, matching the implementation.
- Observation dimensions are written explicitly:
  $d_o^{s}=3N+2$ and $d_o^{u}=5\times3+2=17$.
- The underwater reward now uses the pre-move current
  $\mathbf{u}_{i,k}$ and the projected propulsion command
  $\tilde{\mathbf{a}}_i$; the alignment cosine is defined explicitly.
- The manuscript no longer claims that interpolation never uses future
  information. It now states that this is an offline reanalysis simulation
  using adjacent samples to reconstruct the field at simulated time, and that
  real-time forecast latency is not emulated.
- The energy statement now distinguishes the soft underwater propulsion
  penalty from the surface policy, which has no hard energy budget and uses
  action smoothing without an explicit power term.
- Removed the unused Ray dependency from the reported environment.
- Corrected the underwater implementation to project propulsion commands onto
  the unit disk and to update the five-frame current history every step.
  Smoke testing confirmed 17-dimensional observations, history advancement,
  and a joint action norm no greater than one.
- The old 0/30/50 m final `rC/sb3` weights were moved without deletion to
  `ppo_underwater_fleet/pre_unitdisk_static_history_backup/`; their old figure
  and CSV were also moved to
  `figures/pre_unitdisk_static_history_backup/`.
- The seven-depth completion pipeline now retrains all seven depths under the
  corrected implementation after the current HYCOM current archive finishes.

### Superseded interim 40/60 m seven-depth experiment (2026-10-30)

This section is retained only as development history. Its partial-archive
weights and results are not manuscript evidence.

- Frozen partial archives: 40 m used 200 training TCs and 3,123 segments;
  60 m used 203 training TCs and 3,094 segments.
- Trained ten Active IPPO seeds at 1.5 and 2.0 m/s for both depths in the
  isolated directory `ppo_underwater_fleet_partial/`.
- Re-evaluated all seven depths, Active IPPO, random control, and passive
  drift on the same 23 held-out TCs.
- Produced 14 complete evaluation files in `eval_out_partial/` and the
  aggregate `eval_uwflow_partial_summary.psv`.
- Regenerated the interim Figure 6 with `make_underwater_7depth_preview.py`;
  40 and 60 m are now populated rather than blank.
- Interim Figure 6 ranges: Active IPPO improves mean experienced current speed
  by 1.6--5.6% and path length by 3.7--7.8 times versus passive drift;
  coverage is 3.3--6.2 times greater. Active 2.0 m/s paths are 1.24--1.32
  times longer than 1.5 m/s paths.
- The manuscript labels these 40/60 m results as interim. Complete-archive
  retraining remains mandatory and must replace these values and weights.

### Full-manuscript optimization (2026-10-30)

- Reduced the abstract to 218 words while retaining the surface and subsurface
  quantitative conclusions.
- Defined acronyms at first use and standardized `TC`, `SUV`, and `IPPO`
  terminology.
- Removed the repeated observation-channel bullet list from Section 4.2; the
  complete observation definition now appears once and later sections refer to
  it.
- Corrected the reward-section description of final tracking error to match the
  actual fleet-averaged individual SUV-to-center metric.
- Documented that underwater evaluation uses one central-index segment per
  held-out TC and ten paired deployment seeds.
- Removed duplicated low-pass-filter numbers from the general surface-result
  narrative; the dedicated filtering subsection remains the single source of
  those comparisons.
- Added an explicit physical interpretation of the Figure 6 depth trends:
  experienced current speed and passive drift length decrease with depth as
  the TC-generated current attenuates, whereas active paths decrease only
  moderately because propulsion is depth-independent.
- Clarified that the increasing active/passive path ratio with depth is partly
  a denominator effect and does not imply increasing policy advantage.
- Compressed the conclusion into four paragraphs covering the framework,
  surface findings, subsurface findings, and limitations.
- Consolidated Appendix Table A.2 into grouped rows so that it fits on one
  page without a caption-only or overlap page.

### Abstract and HYCOM mirror update (2026-10-30)

- Rewrote the abstract to 249 words with a clearer problem--method--evidence--
  implication structure, including the surface-to-60 m transit/dive concept,
  upper-ocean temperature--salinity--current observation goal, and the
  requirement for autonomous decisions from environmental observations alone.
- Added `NCSS_BASE` support to the HYCOM downloaders and switched the 40/60 m
  recovery pool to `https://tds.hycom.org/thredds/ncss/grid/` after
  `ncss.hycom.org` began returning persistent HTTP 500/502 and empty replies.
- Added `watch_hycom_tds_40_60.py`, which keeps the 40/60 m tds download pool
  populated until the archive audit reaches 500/500 at both depths.

### Full-description rollback to the 2026-10-30 09:32 PDF

- Restored the complete manuscript prose from the `main(3).pdf` version
  compiled at 2026-10-30 09:32:55 CST.
- The source was reconstructed by replaying the recorded edit history from the
  copied `main_revised_keep_ippo.tex` base and verified against the attached PDF
  text.
- Preserved the later author-list correction: Fangzheng Qi remains the third
  author between Xiwen Ma and Wenlei Peng.
- Saved the superseded prose version as `main_20261030_before_rollback.tex`.
- Consolidated the restored abstract into one paragraph of 249 words while
  retaining the rolled-back manuscript's experiment settings and numerical
  ranges.

### Final complete seven-depth experiment (2026-10-31)

- Completed the 40 and 60 m HYCOM current archives at 500 events each. The
  final missing TDS case, `JANGMI_2014_2014362`, required stitching the
  2014/2015 archive boundary; the resulting 40/60 m files contain the same 39
  time slices as the completed 10/20/30/50 m files.
- Rebuilt all seven 0.25-degree caches from the complete archives; every depth
  contains 500 storm caches.
- Trained 140 corrected final models: seven depths x two propulsion limits x
  ten independent active-policy seeds.
- Evaluated active IPPO, random control, and passive drift on the same 500
  held-out episodes (50 TCs x 10 deployment seeds). The final matrix contains
  14/14 complete evaluation files with 6,000 policy-episode rows each.
- Final active-versus-drift ranges are +2.11% to +5.36% mean experienced
  current speed, 3.63x to 7.27x path length, and 3.25x to 5.70x coverage.
  Every one of the 14 depth/speed configurations has a higher mean current
  speed than drift and random control.
- Active 2.0 m/s paths are 1.24--1.31 times longer than active 1.5 m/s paths.
  Of 140 seed-configuration comparisons, active control wins 133; uncorrected
  paired tests identify 117 significant improvements and three significant
  deficits.
- Replaced all interim/preview fallbacks in `main.tex` with fixed final figure
  inputs. Updated the abstract, underwater results, and conclusion to the
  complete seven-depth ranges and added the propulsion-proxy cost.
- Fixed two aggregation blockers: the underwater table generator is now
  called by `aggregate_underwater_method_ablation.py`, and the final plotting
  script compares speed sets independent of the requested plotting order.
- `verify_uwflow_completion.py` passes with 7 x 500 archives/caches, 140/140
  models, 14/14 evaluations, and all final figures/tables present.
