# Manuscript Source

This directory contains the public manuscript source and generated assets for
the paper.

## Build

```bash
tectonic -X compile --keep-intermediates --keep-logs main.tex
```

Expected output: `main.pdf`.

## Files

| Path | Description |
|---|---|
| `main.tex` | Authoritative manuscript source |
| `main.pdf` | Rendered manuscript |
| `MARL_Ding.bib` | Bibliography |
| `sections/` | Generated numerical results text |
| `tables/` | Generated result tables |
| `figures/` | Manuscript figures |
| `tools/` | Aggregation and figure-generation scripts |

## Generate Results

Run from the repository root:

```bash
python aggregate_surface_vmax_comparison.py
python aggregate_underwater_method_ablation.py
python make_surface_control_ablation_table.py
python Latex_Yihui_MARL/make_fig2_surface_subsurface_observation.py
```

The underwater aggregator rejects incomplete seven-depth data instead of
silently substituting a partial matrix.

## Current Status

- Surface ten-seed results are available.
- Figure 2 is explicitly labeled as interim until final-distance recomputation
  finishes.
- Underwater policies are being retrained after the corrected unit-disk action
  and causal history implementation.
- Preview figures in this directory are not final quantitative evidence.
