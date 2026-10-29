# -*- coding: utf-8 -*-
"""Aggregate the underwater 4-AUV max-flow matrix evaluation into one table.

Reads eval_out/eval_info_info_d*_500.csv (written by run_uwflow_eval.py), parses
depth / vmax / duration out of the file name and algo / reward / seed out of the
policy label, then reports, for every configuration:

    mean|V| at AUV position (the task objective) + paired gain vs idle drift
    coverage area / info coverage / core hit / reconstruction RMSE (P3 family)
    path length, mean speed, p90 flow speed, energy proxy

Usage:  python summarize_uwflow_500.py [glob]
Output: eval_uwflow_500_summary.psv  (+ stdout table)
"""
import os, re, glob, sys
import numpy as np
import pandas as pd
from scipy import stats

BASE = os.path.dirname(os.path.abspath(__file__))
PAT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    BASE, "eval_out", "eval_info_info_d*_500.csv")

METRICS = ["mean_V", "cov_area_km2", "info_cov_pct", "core_hit_pct",
           "recon_rmse", "recon_corr", "path_km", "mean_speed_ms",
           "V_p90", "high_pct", "mean_a2"]


def parse_file(path):
    m = re.search(r"eval_info_info_d(\d+(?:\.\d+)?)_start(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)h_v(\d+(?:\.\d+)?)_500", os.path.basename(path))
    if not m:
        return None
    return dict(depth=float(m.group(1)), start=float(m.group(2)),
                dur=float(m.group(3)), vmax=float(m.group(4)))


def parse_policy(name):
    m = re.match(r"ppo_uwflow_500_d(\d+)_v(\d+\.\d+)_r([A-Z])_(\d+(?:\.\d+)?)h_(\w+)_s(\d+)", name)
    if m:
        return dict(reward=m.group(3), algo=m.group(5), train_seed=int(m.group(6)))
    if name.startswith("idle"):
        return dict(reward="idle", algo="idle", train_seed=-1)
    if name.startswith("random"):
        return dict(reward="random", algo="random", train_seed=-1)
    return dict(reward="?", algo="?", train_seed=-1)


def main():
    files = sorted(glob.glob(PAT))
    if not files:
        print("no files matched:", PAT)
        return
    rows = []
    for f in files:
        meta = parse_file(f)
        if meta is None:
            continue
        df = pd.read_csv(f)
        df["depth"] = meta["depth"]; df["dur"] = meta["dur"]; df["vmax"] = meta["vmax"]
        rows.append(df)
    df = pd.concat(rows, ignore_index=True)
    pol = pd.DataFrame([parse_policy(p) for p in df["policy"]], index=df.index)
    df = pd.concat([df, pol], axis=1)

    out = []
    for (depth, vmax, dur), g in df.groupby(["depth", "vmax", "dur"]):
        piv = {m: g.pivot_table(index=["storm", "seed"], columns="policy", values=m)
               for m in METRICS if m in g.columns}
        for name, gg in g.groupby("policy"):
            info = parse_policy(name)
            rec = dict(depth=depth, vmax=vmax, dur=dur,
                       algo=info["algo"], reward=info["reward"], train_seed=info["train_seed"])
            for m in METRICS:
                if m in gg.columns:
                    rec[m] = float(gg[m].mean())
            # paired gain vs idle on the same (storm, deployment-seed) episodes
            if info["algo"] not in ("idle", "random") and "idle(漂流)" in piv["mean_V"].columns:
                a = piv["mean_V"][name]; b = piv["mean_V"]["idle(漂流)"]
                ok = a.notna() & b.notna()
                if ok.sum() > 2:
                    rec["gain_V_pct"] = float((a[ok].mean() / b[ok].mean() - 1) * 100)
                    rec["p_V"] = float(stats.ttest_rel(a[ok], b[ok]).pvalue)
                if "cov_area_km2" in piv:
                    a2 = piv["cov_area_km2"][name]; b2 = piv["cov_area_km2"]["idle(漂流)"]
                    ok2 = a2.notna() & b2.notna()
                    if ok2.sum() > 2 and b2[ok2].mean() > 0:
                        rec["gain_cov_pct"] = float((a2[ok2].mean() / b2[ok2].mean() - 1) * 100)
                        rec["p_cov"] = float(stats.ttest_rel(a2[ok2], b2[ok2]).pvalue)
            out.append(rec)

    s = pd.DataFrame(out)
    cols = ["depth", "vmax", "dur", "algo", "reward", "train_seed", "mean_V", "gain_V_pct", "p_V",
            "cov_area_km2", "gain_cov_pct", "p_cov", "info_cov_pct", "core_hit_pct",
            "recon_rmse", "recon_corr", "path_km", "mean_speed_ms", "V_p90", "high_pct", "mean_a2"]
    s = s[[c for c in cols if c in s.columns]].sort_values(
        ["depth", "vmax", "dur", "algo", "reward", "train_seed"]).reset_index(drop=True)
    path = os.path.join(BASE, "eval_uwflow_500_summary.psv")
    s.to_csv(path, sep="|", index=False, float_format="%.4f")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 400)
    print(s.to_string(index=False, float_format=lambda x: "%.3f" % x))
    print("\nsaved", path, " rows=%d" % len(s))

    # 3-seed 均值（RL 策略跨训练 seed 的均值±标准差）
    rl = s[s.algo.isin(["ippo", "sb3"])]
    if len(rl):
        agg = rl.groupby(["depth", "vmax", "dur", "algo", "reward"]).agg(
            gain_V=("gain_V_pct", "mean"), gain_V_sd=("gain_V_pct", "std"),
            gain_cov=("gain_cov_pct", "mean"),
            core_hit=("core_hit_pct", "mean"),
            rmse=("recon_rmse", "mean"),
            path=("path_km", "mean"),
        ).reset_index()
        print("\n===== 3-seed 汇总（vs idle 漂流）=====")
        print(agg.to_string(index=False, float_format=lambda x: "%.2f" % x))

    # ---------- markdown 报告（训练全部跑完后直接可用） ----------
    L = []
    L.append("# Underwater 4-AUV max-flow matrix - results\n")
    L.append("Typhoon set: 500 (450 train / 50 test). Start: random deployment within 200 km of the")
    L.append("segment-end eye position. Decision step 10 min. Observation per AUV: own-point flow")
    L.append("(spd/u/v) + 5-frame memory + own velocity = 17 dims. Motion: ground velocity =")
    L.append("propulsion (action x Vmax) + ambient flow.\n")
    L.append("All policies are paired on the same episodes (same segment, same deployment seed);")
    L.append("gains are relative to idle (pure drift). n_seed = deployment seeds per test storm.\n")

    L.append("\n## idle (drift) baselines\n")
    L.append("| depth | vmax | dur | mean\\|V\\| | covered km2 | core hit % | recon RMSE | p90 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for _, r in s[s.algo == "idle"].sort_values(["depth", "vmax", "dur"]).iterrows():
        L.append("| %gm | %.1f | %gh | %.3f | %.0f | %.2f | %.4f | %.3f |"
                 % (r["depth"], r["vmax"], r["dur"], r.get("mean_V", np.nan),
                    r.get("cov_area_km2", np.nan), r.get("core_hit_pct", np.nan),
                    r.get("recon_rmse", np.nan), r.get("V_p90", np.nan)))

    L.append("\n## Trained policies (per training seed)\n")
    L.append("| depth | vmax | dur | algo | reward | seed | mean\\|V\\| | gain V % | p | cover gain % | p | core hit % | RMSE | path km | energy |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in s[s.algo.isin(["ippo", "sb3"])].sort_values(
            ["depth", "vmax", "dur", "algo", "reward", "train_seed"]).iterrows():
        L.append("| %gm | %.1f | %gh | %s | %s | %d | %.3f | %+.2f | %.3f | %+.1f | %.3f | %.2f | %.4f | %.1f | %.3f |"
                 % (r["depth"], r["vmax"], r["dur"], r["algo"], r["reward"], r["train_seed"],
                    r.get("mean_V", np.nan), r.get("gain_V_pct", np.nan), r.get("p_V", np.nan),
                    r.get("gain_cov_pct", np.nan), r.get("p_cov", np.nan),
                    r.get("core_hit_pct", np.nan), r.get("recon_rmse", np.nan),
                    r.get("path_km", np.nan), r.get("mean_a2", np.nan)))

    if len(rl):
        L.append("\n## Aggregated over training seeds\n")
        L.append("| depth | vmax | dur | algo | reward | n | gain V % (mean+-sd) | cover gain % | core hit % | RMSE | path km | energy |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        agg2 = rl.groupby(["depth", "vmax", "dur", "algo", "reward"]).agg(
            n=("gain_V_pct", "size"), gV=("gain_V_pct", "mean"), gVsd=("gain_V_pct", "std"),
            gC=("gain_cov_pct", "mean"), core=("core_hit_pct", "mean"),
            rmse=("recon_rmse", "mean"), path=("path_km", "mean"),
            energy=("mean_a2", "mean")).reset_index()
        for _, r in agg2.sort_values(["depth", "vmax", "dur", "algo", "reward"]).iterrows():
            L.append("| %gm | %.1f | %gh | %s | %s | %d | %+.2f +- %.2f | %+.1f | %.2f | %.4f | %.1f | %.3f |"
                     % (r["depth"], r["vmax"], r["dur"], r["algo"], r["reward"], r["n"],
                        r["gV"], 0.0 if pd.isna(r["gVsd"]) else r["gVsd"], r["gC"], r["core"],
                        r["rmse"], r["path"], r["energy"]))

    mdp = os.path.join(BASE, "eval_uwflow_500_report.md")
    open(mdp, "w").write("\n".join(L) + "\n")
    print("saved", mdp)


if __name__ == "__main__":
    main()
