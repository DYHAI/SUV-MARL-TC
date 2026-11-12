# Final seven-depth underwater comparison

All policies are paired on the same 500 held-out episodes (50 TCs x 10 deployment seeds). Active values are means over ten independent training seeds. Random and passive drift are untrained paired baselines. Current and coverage columns use Active/Random/Drift ordering. Weighted coverage is reported in $10^3$ m s$^{-1}$ km$^2$.

| Depth | Vmax | Current A/R/D (m/s) | Active gain vs drift | Coverage A/R/D (km2) | Weighted coverage A/R/D | Seed wins | Significant + / - |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 m | 1.5 m/s | 0.532/0.522/0.521 | +2.11% | 3527/1323/1084 | 2.012/0.823/0.701 | 9/10 | 8/1 |
| 0 m | 2.0 m/s | 0.544/0.522/0.521 | +4.27% | 4386/1474/1084 | 2.319/0.890/0.701 | 10/10 | 10/0 |
| 10 m | 1.5 m/s | 0.471/0.460/0.460 | +2.43% | 3417/1208/969 | 1.744/0.658/0.551 | 10/10 | 8/0 |
| 10 m | 2.0 m/s | 0.475/0.460/0.460 | +3.32% | 4172/1349/969 | 2.107/0.718/0.551 | 10/10 | 10/0 |
| 20 m | 1.5 m/s | 0.446/0.432/0.432 | +3.23% | 3449/1159/921 | 1.660/0.603/0.503 | 10/10 | 9/0 |
| 20 m | 2.0 m/s | 0.447/0.432/0.432 | +3.53% | 4241/1307/921 | 1.956/0.660/0.503 | 9/10 | 9/0 |
| 30 m | 1.5 m/s | 0.409/0.398/0.398 | +2.69% | 3221/1097/850 | 1.481/0.533/0.441 | 10/10 | 7/0 |
| 30 m | 2.0 m/s | 0.408/0.398/0.398 | +2.52% | 3995/1238/850 | 1.750/0.586/0.441 | 9/10 | 7/1 |
| 40 m | 1.5 m/s | 0.366/0.356/0.356 | +2.72% | 3147/1016/765 | 1.191/0.446/0.359 | 9/10 | 8/1 |
| 40 m | 2.0 m/s | 0.366/0.357/0.356 | +2.78% | 4066/1156/765 | 1.555/0.494/0.359 | 8/10 | 7/0 |
| 50 m | 1.5 m/s | 0.325/0.317/0.316 | +2.85% | 2980/932/680 | 1.069/0.359/0.285 | 9/10 | 7/0 |
| 50 m | 2.0 m/s | 0.333/0.317/0.316 | +5.36% | 3771/1072/680 | 1.374/0.404/0.285 | 10/10 | 10/0 |
| 60 m | 1.5 m/s | 0.301/0.292/0.292 | +3.19% | 2905/879/622 | 0.959/0.310/0.242 | 10/10 | 8/0 |
| 60 m | 2.0 m/s | 0.306/0.293/0.292 | +4.92% | 3546/1014/622 | 1.207/0.347/0.242 | 10/10 | 9/0 |

## Summary

- Active mean-current gain versus drift: +2.11% to +5.36%.
- Active nominal coverage is 3.25 to 5.70 times that of drift.
- Active current-weighted coverage is 2.87 to 4.99 times that of drift.
- Active mean-current gain versus random control: +2.01% to +5.13%.
- Positive active-current gain appears in 14/14 configurations.
- Active wins in 133/140 seed-configuration comparisons.
