# Citation and full-text audit (2026-11-01)

## Scope

- Manuscript: `main.tex`
- Bibliography: `MARL_Ding.bib`
- Final PDF: `main.pdf`
- Reference entries: 59
- Citation commands: 32
- Every bibliography entry is cited; there are no missing or unused keys.
- No duplicate DOI or duplicate title was found.
- All 57 entries with a DOI resolve. The 57 DOI records were also matched
  against OpenAlex/Crossref title and year metadata. `r44` and `r49` have no
  DOI and were checked as conference proceedings.

## Reference-by-reference audit

| Entry | Use in manuscript | Verdict |
|---|---|---|
| r1 | Detection and attribution of TC changes, including observational uncertainty | Correct after the sentence was split from projections. |
| r2 | Projected increase in the proportion of very intense TCs | Correct. |
| r3 | Possible decline in global TC frequency | Correct. |
| r4 | Increase in offshore rapid-intensification events | Correct. |
| r5 | Intensification of northwest-Pacific landfalling typhoons | Correct. |
| r6 | Increasing destructive potential of landfalling TCs over China | Correct. |
| r7 | Increasing nearshore TC intensification | Correct. |
| r8 | Western North Pacific TC exposure | Correct. |
| r9 | NHC progress in TC intensity forecasting | Correct. |
| r10 | Errors and uncertainty in western-North-Pacific intensity forecasts | Correct. |
| r11 | Review of upper-ocean response to TCs | Correct. |
| r12 | Barrier layers and TC intensification | Correct. |
| r13 | Ocean observations supporting TC studies and forecasts | Correct. |
| r14 | Aircraft/dropsonde near-core observation campaign | Correct. |
| r15 | Small-unmanned-aircraft observations in hurricane eyewalls | Correct. |
| r16 | Long-duration underwater-glider observations | Correct; “slower than a translating TC” is a valid limitation. |
| r17 | Adaptive AUV sampling | Correct. |
| r18 | Underwater-glider data assimilation for Hurricane Gonzalo | Correct. |
| r19 | OSSE assessment of underwater-glider arrays | Correct. |
| r20 | Coupled atmosphere--ocean ensemble data assimilation | Correct. |
| r21 | Strongly coupled ocean-current assimilation for Typhoon Hato | Correct. |
| r22 | Satellite and glider assimilation for Hurricane Isaias | Correct. |
| r23 | Targeted ocean sampling and sensitivity to initial ocean conditions | Correct. |
| r24 | CNOP-based sensitive areas for targeted TC observation | Correct after changing “ocean observations” to the broader “targeted-observation studies.” |
| r25 | Adjoint-derived sensitive areas for TC movement | Correct after the same wording change. |
| r26 | AUV TC center tracking using forecast-path correction | Correct. |
| r27 | Near-surface active observation planning for TC conditions | Correct. The fleet-size-scaling statement was narrowed to what the paper did not report. |
| r28 | Blue Whale submersible unmanned vessel | Bibliographic record verified. The supplied article confirms the 23 kn surface speed, 3 kn submerged speed, ≥4 h submerged operation, ≥72 h standby, and ≥800 kg payload. The approximately 60 m public-reference depth is retained at the author's direction. |
| r29 | Wave-propelled semi-submersible unmanned vehicle | Correct as part of the SUV platform class. |
| r30 | Experimentally validated depth control of a semi-submersible vehicle | Correct as part of the SUV platform class. |
| r31 | Rocketsonde launched from an unmanned semi-submersible vehicle | Correct as a platform-capability example. |
| r32 | Multi-AUV control and adaptive sampling | Correct. |
| r33 | RL-based tracking of an underwater target | Correct. |
| r34 | Multi-AUV cooperative multi-target tracking with MARL | Correct. |
| r35 | Cooperative multi-AUV hunting under weak information | Correct. |
| r36 | Multi-AUV cooperative search with acoustic--optical communication and DRL | Correct. |
| r37 | ERA5 reanalysis | Correct. |
| r38 | HYCOM data-assimilative system | Correct. |
| r39 | IBTrACS best-track archive | Correct; OpenAlex lists the online year as 2009, while the bibliography uses the formal 2010 journal issue. |
| r40 | PPO algorithm | Correct. |
| r41 | Independent learning in cooperative multi-agent tasks | Correct. |
| r42 | Effectiveness of PPO in cooperative multi-agent games | Correct. |
| r43 | A* graph search | Correct. |
| r44 | D* Lite replanning | Correct; no DOI, conference bibliographic data checked separately. |
| r45 | Sampling-based optimal motion planning | Correct. |
| r46 | Artificial-potential-field obstacle avoidance | Correct. |
| r47 | Constrained model predictive control | Correct. |
| r48 | AUV path-planning review | Correct. |
| r49 | Deep RL from human preferences | Correct; no DOI, NeurIPS bibliographic data checked separately. |
| r50 | RL from human feedback for language models | Correct in the statement that RL spans diverse domains. |
| r51 | RL for LLM reasoning | Correct in the same broad-domain statement. |
| r52 | RL for agile legged robots | Correct. |
| r53 | RL for dexterous manipulation | Correct. |
| r54 | RL for champion-level drone racing | Correct. |
| r55 | Multi-agent RL for StarCraft II | Correct. |
| r56 | Large-scale RL for Dota 2 | Correct. |
| r57 | StarCraft Multi-Agent Challenge | Correct for cooperative multi-agent evaluation. |
| r58 | Active scatterometer limitations, rain contamination, and high-wind retrieval | Correct. The manuscript now separates active scatterometers from passive radiometers. |
| r59 | SFMR passive-microwave wind bias in heavy precipitation | Correct. The text now limits the strongest error statement to heavy rain and lower wind speeds. |

## Substantive full-text corrections

1. Split the observed/projected TC trend citations instead of treating r1--r4
   as supporting one combined claim.
2. Removed the unsupported claim that data-driven models improved track
   forecasting; the track/intensity statement now follows r9--r10 directly.
3. Separated active scatterometer and passive radiometer limitations.
4. Separated targeted-ocean-sampling evidence from atmospheric targeted-
   observation evidence.
5. Replaced “active control consistently outperforms” with the exact evidence:
   positive configuration means in all 14 depth-speed cells, 133/140 positive
   seed-level comparisons, 117 significant improvements, and 3 significant
   deficits.
6. Reclassified the 20 h underwater period as a simulation/evaluation horizon,
   not a battery-endurance claim. The public-reference endurance value remains
   separate in Table 2.
7. Corrected the subsurface initialization description: the separate
   subsurface episode is aligned to the surface segment end time and samples
   within 200 km of the TC center; it does not inherit surface endpoint
   coordinates.
8. Corrected the coastal-safety scope. Offshore rejection and action safety
   filtering are surface-only; the underwater matrix retains the nearshore
   penalty and terminates at landfall.
9. Corrected the 200 km circle in the Figure 5 caption. It is the initialization
   boundary; open circles are initial vehicle positions.
10. Corrected the coverage definition to a nominal 0.05-degree grid-cell area
    approximation.
11. Corrected the table’s former “energy proxy.” The evaluation metric is the
    squared norm of the unprojected policy action, so it is now labelled a
    command/control-effort proxy, not calibrated propulsion energy.
12. Removed Ray from the software list because it is not used by the final
    reported experiments.
13. Corrected the spatial domain statement: 98.7E--171.6E and 6.8N--56.7N is
    the archive-selection box; individual event crops can extend beyond it.
14. Replaced an unsupported broad statement about current-speed quantiles with
    quantities directly supported by the final result table and figure.
15. Revised the direct-pursuit discussion: even 100--200 km offsets can require
    more than a typical 20 h episode at the 12 m/s limit; neither speed limit is
    described as guaranteeing convergence.
16. Kept the ten-seed underwater numerical table out of the manuscript at the
    author's request; Figure 6 remains the paper-facing three-metric display.

## Numerical checks against processed results

- 500 TCs; 495 with usable segments; 7,248 segments.
- 450 training TCs; 445 usable; 6,480 training segments.
- 50 test TCs; 768 test segments; no TC overlap.
- Realized segment length: 292.1--507.1 km; median 396.4 km.
- Duration: 2.2--103 h; median 16.5 h; mean 20.0 h.
- Surface final errors reproduce the values stated in the manuscript for both
  speed limits and all five deployment radii.
- Underwater outputs:
  - 14/14 positive configuration means versus passive drift.
  - 133/140 positive seed-level comparisons.
  - 117 significant improvements and 3 significant deficits.
  - Current-speed gain: 2.11%--5.36%.
  - Path ratio: 3.63--7.27.
  - Nominal coverage ratio: 3.25--5.70.
- All 14 final underwater evaluation files contain 6,000 rows:
  50 TCs x 10 deployment seeds x 12 policies.
- `verify_uwflow_completion.py`: passed.
- Final manuscript: 21 pages, 59 references, no undefined citations and no
  BibTeX warnings.

## Remaining limitations that are disclosed rather than “fixed”

- The study remains simulation-based and uses a horizontal point-mass model.
- The `r28` Blue Whale engineering values could not be independently accessed
  through the publisher's Cloudflare-protected page during this audit; the DOI
  and bibliographic record resolve, and all numerical platform scenarios are
  explicitly labelled assumptions rather than certified data.
- The 20 h comparison horizon exceeds the public-reference submerged-endurance
  value; the manuscript now states this explicitly and does not claim that 20 h
  endurance has been demonstrated.
- Underwater initialization does not use the surface offshore-rejection
  procedure, and underwater safety is based on penalty plus landfall
  termination rather than the surface action filter.
