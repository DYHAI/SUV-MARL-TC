# Final seven-depth underwater comparison

All policies are paired on the same 500 held-out episodes (50 TCs x 10 deployment seeds). Active values are means over ten independent training seeds. Random and passive drift are untrained paired baselines.

| Depth | Vmax | Current A/R/D (m/s) | Active gain vs drift | Coverage A/R/D (km2) | Path A/D | Energy A/R/D | Seed wins | Significant + / - |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 m | 1.5 m/s | 0.532/0.522/0.521 | +2.11% | 3527/1323/1084 | 3.63x | 1.487/0.664/0.000 | 9/10 | 8/1 |
| 0 m | 2.0 m/s | 0.544/0.522/0.521 | +4.27% | 4386/1474/1084 | 4.61x | 1.532/0.664/0.000 | 10/10 | 10/0 |
| 10 m | 1.5 m/s | 0.471/0.460/0.460 | +2.43% | 3417/1208/969 | 4.03x | 1.502/0.663/0.000 | 10/10 | 8/0 |
| 10 m | 2.0 m/s | 0.475/0.460/0.460 | +3.32% | 4172/1349/969 | 5.01x | 1.440/0.663/0.000 | 10/10 | 10/0 |
| 20 m | 1.5 m/s | 0.446/0.432/0.432 | +3.23% | 3449/1159/921 | 4.32x | 1.529/0.663/0.000 | 10/10 | 9/0 |
| 20 m | 2.0 m/s | 0.447/0.432/0.432 | +3.53% | 4241/1307/921 | 5.40x | 1.525/0.663/0.000 | 9/10 | 9/0 |
| 30 m | 1.5 m/s | 0.409/0.398/0.398 | +2.69% | 3221/1097/850 | 4.46x | 1.478/0.663/0.000 | 10/10 | 7/0 |
| 30 m | 2.0 m/s | 0.408/0.398/0.398 | +2.52% | 3995/1238/850 | 5.59x | 1.470/0.663/0.000 | 9/10 | 7/1 |
| 40 m | 1.5 m/s | 0.366/0.356/0.356 | +2.72% | 3147/1016/765 | 4.84x | 1.471/0.663/0.000 | 9/10 | 8/1 |
| 40 m | 2.0 m/s | 0.366/0.357/0.356 | +2.78% | 4066/1156/765 | 6.35x | 1.604/0.663/0.000 | 8/10 | 7/0 |
| 50 m | 1.5 m/s | 0.325/0.317/0.316 | +2.85% | 2980/932/680 | 5.30x | 1.439/0.663/0.000 | 9/10 | 7/0 |
| 50 m | 2.0 m/s | 0.333/0.317/0.316 | +5.36% | 3771/1072/680 | 6.96x | 1.499/0.663/0.000 | 10/10 | 10/0 |
| 60 m | 1.5 m/s | 0.301/0.292/0.292 | +3.19% | 2905/879/622 | 5.76x | 1.491/0.663/0.000 | 10/10 | 8/0 |
| 60 m | 2.0 m/s | 0.306/0.293/0.292 | +4.92% | 3546/1014/622 | 7.27x | 1.440/0.663/0.000 | 10/10 | 9/0 |

## Summary

- Active mean-current gain vs drift: +2.11% to +5.36%.
- Active ground path / drift ground path: 3.63x to 7.27x.
- Active coverage / drift coverage: 3.25x to 5.70x.
- Active mean-current gain vs random control: +2.01% to +5.13%.
- Positive active-current gain in 14/14 configurations.
- Active wins in 133/140 seed-configuration comparisons.
