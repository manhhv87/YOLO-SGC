"""Accuracy summaries and statistical tests reported in the paper.

Everything is computed from the per-run result files in paper_results/:
  table1_results.json                Tables 1 and 5, Section 4.1
  seed5_full.json                    Table 2, Section 4.2
  phaseB_results.json                Table 3, Section 4.3
  scalemap_intervention.json,
  scalemap_intervention2.json,
  interv_fusion_main.json            Table 4
  interv_allruns/                    Section 4.4 (all forty SGC models)
  deploy_allruns.json                Table 6 (accuracy column), Section 4.6
  prune5_results.json                Table 7, Section 4.7
  guidance_seeds.json,
  guidance_bands.json                Table 8, Section 4.9

Tables use the 0-1 scale; differences are in points (x100). Welch tests are two-sided
and uncorrected unless stated otherwise. Requires numpy and scipy >= 1.11.

Usage:
  python stats.py [--results paper_results]
"""
import argparse
import json
import os

import numpy as np
from scipy import stats
from scipy.stats import nct

BACKBONES = ["yolov5s", "yolov8s", "yolov9s", "yolov11s"]
VARIANTS = ["_sgc_p345", "_sgc_p45"]
BB_NAME = {"yolov5s": "YOLOv5s", "yolov8s": "YOLOv8s", "yolov9s": "YOLOv9s", "yolov11s": "YOLO11s"}
CLASSES = ["near", "mid_near", "mid_far", "far"]


def name(cfg):
    bb = next(b for b in BACKBONES if cfg.startswith(b))
    return BB_NAME[bb] + cfg[len(bb):].replace("_sgc_p", "-SGC-P")


def values(runs, metric):
    """One metric over the runs of a configuration, 0-1 scale."""
    out = []
    for r in runs.values():
        if metric in r:
            out.append(r[metric])
        else:
            out.append((r.get("per") or r["per_class_ap50"])[metric])
    return np.array(out, float)


def ms(x):
    return f"{x.mean():.3f}+/-{x.std(ddof=1):.3f}"


def welch(a, b):
    return stats.ttest_ind(a, b, equal_var=False).pvalue


def fp(p):
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


def hedges_g(a, b):
    n1, n2 = len(a), len(b)
    sp = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    g = (a.mean() - b.mean()) / sp * (1 - 3 / (4 * (n1 + n2) - 9))
    se = np.sqrt((n1 + n2) / (n1 * n2) + g ** 2 / (2 * (n1 + n2)))
    return g, g - 1.96 * se, g + 1.96 * se


def holm(pvals):
    p = np.asarray(pvals, float)
    adj = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(np.argsort(p)):
        running = max(running, min(1.0, p[idx] * (len(p) - rank)))
        adj[idx] = running
    return adj


def pooled_difference(pairs):
    """Mean of (a - b) over backbones from a two-way backbone x neck model with pooled variance.

    pairs: list of (a, b) arrays in points, one pair per backbone. Returns diff, CI low, CI high, p.
    """
    groups = [g for pair in pairs for g in pair]
    n = len(groups[0])
    resid = np.concatenate([g - g.mean() for g in groups])
    df = len(resid) - len(groups)
    s2 = (resid ** 2).sum() / df
    se = np.sqrt(s2 * (2 / n) / len(pairs))
    md = float(np.mean([a.mean() - b.mean() for a, b in pairs]))
    q = stats.t.ppf(0.975, df)
    return md, md - q * se, md + q * se, 2 * stats.t.sf(abs(md / se), df)


def mde_sd_units(n=5, alpha=0.05, power=0.8):
    """Smallest standardized difference detectable by a two-sample t test (noncentral t)."""
    df = 2 * n - 2
    tc = stats.t.ppf(1 - alpha / 2, df)

    def pw(d):
        ncp = d / np.sqrt(2 / n)
        return 1 - nct.cdf(tc, df, ncp) + nct.cdf(-tc, df, ncp)

    lo, hi = 0.01, 10.0
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if pw(mid) < power else (lo, mid)
    return hi


def dunnett(samples, control, seed=0):
    """Dunnett many-to-one p-values; the seed fixes scipy's numerical integration."""
    try:
        res = stats.dunnett(*samples, control=control, rng=np.random.default_rng(seed))
    except TypeError:
        res = stats.dunnett(*samples, control=control, random_state=np.random.default_rng(seed))
    return res.pvalue


def load(results, fname):
    with open(os.path.join(results, fname)) as f:
        return json.load(f)


def table1(t1):
    print("== Table 1: before pruning, five runs per row (mean +/- SD)")
    print(f"  {'Model':18s} {'mAP@50':>13s} {'mAP@50-95':>13s} {'far-class AP':>13s}")
    for bb in BACKBONES:
        for cfg in [bb] + [bb + v for v in VARIANTS]:
            r = t1[cfg]
            print(f"  {name(cfg):18s} {ms(values(r, 'mAP50')):>13s} {ms(values(r, 'mAP50_95')):>13s} "
                  f"{ms(values(r, 'far')):>13s}")

    print("\n== Table 5: per-class AP@50 by distance band, five runs per row")
    print(f"  {'Model':18s} " + " ".join(f"{c:>13s}" for c in CLASSES + ["mAP@50"]))
    for bb in BACKBONES:
        for cfg in (bb, bb + "_sgc_p345"):
            r = t1[cfg]
            print(f"  {name(cfg):18s} " + " ".join(f"{ms(values(r, m)):>13s}" for m in CLASSES + ["mAP50"]))


def section_4_1(t1):
    print("\n== Section 4.1: SGC minus its baseline (points); Welch p, Holm over the eight variants, Hedges' g")
    for m, label in (("mAP50", "mAP@50"), ("far", "far-class AP")):
        rows = []
        for bb in BACKBONES:
            b = values(t1[bb], m) * 100
            for v in VARIANTS:
                a = values(t1[bb + v], m) * 100
                rows.append((name(bb + v), a.mean() - b.mean(), welch(a, b), *hedges_g(a, b)))
        adj = holm([r[2] for r in rows])
        print(f"  -- {label}")
        for r, h in zip(rows, adj):
            print(f"     {r[0]:18s} {r[1]:+6.2f}  p={r[2]:.3f}  Holm p={h:.3f}  g={r[3]:+.2f} [{r[4]:+.2f}, {r[5]:+.2f}]")
        print(f"     smallest Holm-adjusted p: {adj.min():.3f}")

    print("  -- Average over the four backbones (two-way backbone x neck model, pooled variance)")
    for v in VARIANTS:
        for m, label in (("mAP50", "mAP@50"), ("far", "far-class AP")):
            pairs = [(values(t1[bb + v], m) * 100, values(t1[bb], m) * 100) for bb in BACKBONES]
            md, lo, hi, p = pooled_difference(pairs)
            print(f"     SGC-P{v[6:]:4s} {label:13s} {md:+.2f} [{lo:.2f}, {hi:.2f}]  p={p:.5f}")

    cfgs = [bb + v for bb in BACKBONES for v in [""] + VARIANTS]
    for m, label in (("mAP50", "mAP@50"), ("far", "far-class AP")):
        sd = [values(t1[c], m).std(ddof=1) * 100 for c in cfgs]
        print(f"  Inter-seed SD of {label} over the twelve configurations: {min(sd):.1f} to {max(sd):.1f} points")
    v8 = [values(t1[c], "mAP50") * 100 for c in ("yolov8s", "yolov8s_sgc_p345", "yolov8s_sgc_p45")]
    sd8 = np.sqrt(np.mean([x.var(ddof=1) for x in v8]))
    d = mde_sd_units()
    print(f"  Minimum detectable difference, five runs per arm, 80% power: {d:.2f} SD; "
          f"at the YOLOv8s-family SD of mAP@50 ({sd8:.2f} points): {d * sd8:.2f} points")


def table2(nk):
    print("\n== Table 2: necks on YOLOv8s, five runs per row")
    order = [("YOLOv8s", "FPN/PAN (baseline)"), ("BiFPN", "BiFPN"), ("AFPN", "AFPN"),
             ("Drop-P3", "Drop-P3 (no SGC)"), ("SGC-P45", "SGC-P45"), ("SGC-P345", "SGC-P345")]
    print(f"  {'Neck':18s} {'mAP@50':>13s} {'mAP@50-95':>13s} {'far-class AP':>13s}")
    arr = {k: {m: np.array(nk[k][m], float) for m in ("map", "map95", "far")} for k, _ in order}
    for k, label in order:
        print(f"  {label:18s} " + " ".join(f"{ms(arr[k][m]):>13s}" for m in ("map", "map95", "far")))
    print("  SGC-P345 minus each neck (points), Welch p:")
    ref = arr["SGC-P345"]
    for k, label in order[:3]:
        print(f"     {label:18s} mAP@50 {100 * (ref['map'].mean() - arr[k]['map'].mean()):+.2f} "
              f"({fp(welch(ref['map'], arr[k]['map']))})  far {100 * (ref['far'].mean() - arr[k]['far'].mean()):+.2f} "
              f"({fp(welch(ref['far'], arr[k]['far']))})")


def table3(pb):
    rows = [("Baseline", "YOLOv8s (FPN)"), ("AMRF_FPN", "AMRF+FPN"), ("Minimal", "Minimal"),
            ("NoAMRF", "w/o AMRF"), ("NoScale", "w/o ScaleMap"), ("EqualWeight", "w/o Adpt. W."),
            ("Full", "Full SGCBlock")]
    print("\n== Table 3: ablation on YOLOv8s-SGC-P345, five runs per row; Welch p against Full")
    v = {k: {m: values(pb[k], m) for m in ("mAP50", "far")} for k in [r[0] for r in rows] + ["YPrior"]}
    full = v["Full"]
    print(f"  {'Variant':18s} {'mAP@50':>13s} {'far-class AP':>13s}  p (mAP@50)")
    for k, label in rows:
        p = "" if k == "Full" else f"  {welch(v[k]['mAP50'], full['mAP50']):.3f}"
        print(f"  {label:18s} {ms(v[k]['mAP50']):>13s} {ms(v[k]['far']):>13s}{p}")

    def diff(a, b, m):
        return 100 * (v[a][m].mean() - v[b][m].mean()), welch(v[a][m], v[b][m])

    print("\n== Section 4.3: contrasts (points, uncorrected Welch p)")
    for a, b, far_p in (("Full", "Baseline", True), ("Minimal", "Full", False), ("AMRF_FPN", "Baseline", False),
                        ("NoAMRF", "Full", False), ("AMRF_FPN", "Full", True), ("NoScale", "Full", False)):
        (dm, pm), (dfar, pfar) = diff(a, b, "mAP50"), diff(a, b, "far")
        print(f"  {a:9s} - {b:9s} mAP@50 {dm:+.2f} (p={pm:.3f})  far {dfar:+.2f}" + (f" (p={pfar:.3f})" if far_p else ""))
    print(f"  Fixed vertical-gradient prior: mAP@50 {ms(v['YPrior']['mAP50'])}, far {ms(v['YPrior']['far'])}")
    for b in ("Full", "NoScale"):
        print(f"     against {b:8s} p={welch(v['YPrior']['mAP50'], v[b]['mAP50']):.3f} (mAP@50), "
              f"p={welch(v['YPrior']['far'], v[b]['far']):.3f} (far)")
    arms = [r[0] for r in rows if r[0] != "Full"] + ["YPrior"]
    print("  Dunnett against Full (Table 3 rows and the fixed prior):")
    for m, label in (("mAP50", "mAP@50"), ("far", "far-class AP")):
        p = dunnett([v[k][m] for k in arms], full[m])
        print(f"     {label:13s} " + "  ".join(f"{k} {x:.3f}" for k, x in zip(arms, p)))


def table4(results):
    print("\n== Table 4: interventions on the seed-0 YOLOv8s-SGC-P345 (points against the unmodified model)")
    rows = [("scalemap_intervention2.json", "zero", "Scale map set to zero"),
            ("scalemap_intervention.json", "const", "Scale map replaced by its spatial mean"),
            ("scalemap_intervention.json", "vflip", "Scale map flipped vertically"),
            ("scalemap_intervention2.json", "shuffle", "Scale map rows randomly permuted"),
            ("scalemap_intervention.json", "ramp", "Scale map replaced by a linear ramp"),
            ("interv_fusion_main.json", "uniform", "Fusion weights forced uniform"),
            ("interv_fusion_main.json", "zmean", "Fusion weights, spatial variation removed"),
            ("interv_fusion_main.json", "zshuffle", "Fusion weights permuted spatially")]
    base = load(results, "interv_fusion_main.json")["none"]
    print(f"  unmodified model: mAP@50 {base['mAP50']:.3f}, far-class AP {base['per_class_ap50']['far']:.3f}")
    worst = 0.0
    for fname, mode, label in rows:
        d = load(results, fname)
        r, b = d[mode], d["none"]
        dm = 100 * (r["mAP50"] - b["mAP50"])
        dfar = 100 * (r["per_class_ap50"]["far"] - b["per_class_ap50"]["far"])
        print(f"  {label:42s} {dm:+.2f}  {dfar:+.2f}")
        if fname.startswith("scalemap"):
            worst = max([worst, abs(dm)] + [abs(100 * (r["per_class_ap50"][c] - b["per_class_ap50"][c])) for c in CLASSES])
    print(f"  scale-map interventions: largest change of mAP@50 or any class AP {worst:.2f} points")


def section_4_4(results):
    print("\n== Section 4.4: interventions on all forty trained SGC models (points)")
    d = {}
    folder = os.path.join(results, "interv_allruns")
    for f in sorted(os.listdir(folder)):
        stem, target = f[:-5].rsplit("_", 1)
        cfg = stem.rsplit("_s", 1)[0]
        r = load(folder, f)
        for mode, x in r.items():
            if mode != "none":
                d.setdefault((target, mode), {}).setdefault(cfg, []).append(
                    (100 * (x["mAP50"] - r["none"]["mAP50"]),
                     100 * (x["per_class_ap50"]["far"] - r["none"]["per_class_ap50"]["far"])))

    def collect(modes):
        return np.array([x for (t, m), by in d.items() if t == "scalemap" and m in modes for v in by.values() for x in v])

    a = collect(("zero", "const", "vflip", "shuffle", "ramp"))
    print(f"  scale map (zero, mean, flip, permutation, ramp): {len(a)} model-interventions, "
          f"max |d mAP@50| {np.abs(a[:, 0]).max():.2f}, max |d far| {np.abs(a[:, 1]).max():.2f}")
    a = collect(("zero", "vflip", "shuffle"))
    print(f"  scale map (zero, flip, permutation): max |d mAP@50| {np.abs(a[:, 0]).max():.2f}")
    by = d[("fusion", "uniform")]
    m = [np.mean([x[0] for x in v]) for v in by.values()]
    f = [np.mean([x[1] for x in v]) for v in by.values()]
    print(f"  fusion weights forced uniform, mean per configuration: mAP@50 {max(m):+.2f} to {min(m):+.2f}, "
          f"far {max(f):+.2f} to {min(f):+.2f}")


def table6(t1, dep):
    print("\n== Table 6 (accuracy column) and Section 4.6: deleting the prior branch, deployed minus trained mAP@50")
    deltas = {}
    for k, v in dep.items():
        cfg, seed = k.rsplit("_s", 1)
        deltas[(cfg, int(seed))] = 100 * (v["mAP50"] - t1[cfg][v["src"]]["mAP50"])
    for bb in BACKBONES:
        for var in VARIANTS:
            print(f"  {name(bb + var):18s} seed 0: {deltas[(bb + var, 0)]:+.2f}")
    print(f"  all {len(deltas)} trained SGC models: max |d mAP@50| {max(abs(x) for x in deltas.values()):.2f}")


def table7(t1, p5):
    print("\n== Table 7: after structured pruning, five runs per row (mean +/- SD)")
    print(f"  {'Model':18s} {'mAP@50':>13s} {'mAP@50-95':>13s}")
    for bb in BACKBONES:
        for cfg in [bb] + [bb + v for v in VARIANTS]:
            print(f"  {name(cfg):18s} {ms(values(p5[cfg], 'mAP50')):>13s} {ms(values(p5[cfg], 'mAP50_95')):>13s}")
    print("\n== Section 4.7")
    ch = []
    for cfg in p5:
        a, b = values(p5[cfg], "mAP50") * 100, values(t1[cfg], "mAP50") * 100
        ch.append((a.mean() - b.mean(), welch(a, b)))
    print(f"  pruned minus unpruned mAP@50: {min(c[0] for c in ch):+.2f} to {max(c[0] for c in ch):+.2f} points, "
          f"smallest Welch p {min(c[1] for c in ch):.3f}")
    pairs = [(values(p5[bb + "_sgc_p45"], "mAP50") * 100, values(p5[bb], "mAP50") * 100) for bb in BACKBONES]
    md, lo, hi, p = pooled_difference(pairs)
    print(f"  pruned SGC-P45 minus pruned baseline, average over backbones: {md:+.2f} [{lo:.2f}, {hi:.2f}] points, p={p:.4f}")


def table8(gd, gb):
    print("\n== Table 8: guidance line on the test set; learned pipelines over five trained models")
    c = gd["classical"]
    print(f"  {'Classical (ExG)':18s} angle {c['ang_mean']:.2f} / {c['ang_med']:.2f} deg  far offset {c['far_mean']:.3f}")
    vals = {}
    for cfg in ("yolov8s", "yolov8s_sgc_p345", "yolov8s_sgc_p45"):
        vals[cfg] = {k: np.array([r[k] for r in gd[cfg].values()]) for k in ("ang_mean", "ang_med", "far_mean", "four")}
        v = vals[cfg]
        print(f"  {name(cfg):18s} angle {v['ang_mean'].mean():.2f}+/-{v['ang_mean'].std(ddof=1):.2f} / "
              f"{v['ang_med'].mean():.2f} deg  far offset {v['far_mean'].mean():.3f}  "
              f"four-band {v['four'].mean():.1f}+/-{v['four'].std(ddof=1):.1f}%")
    print("\n== Section 4.9")
    ps = []
    for cfg in ("yolov8s_sgc_p345", "yolov8s_sgc_p45"):
        for k in ("ang_mean", "four"):
            ps.append(welch(vals[cfg][k], vals["yolov8s"][k]))
    print(f"  SGC pipelines against YOLOv8s, angle and four-band rate: smallest Welch p {min(ps):.3f}")
    ra = [c["ang_mean"] / v["ang_mean"].mean() for v in vals.values()]
    rf = [c["far_mean"] / v["far_mean"].mean() for v in vals.values()]
    print(f"  classical / learned: angle {min(ra):.2f}x to {max(ra):.2f}x, far offset {min(rf):.1f}x to {max(rf):.1f}x")
    for band in ("near", "far"):
        loo = [np.mean([r["bands"][band]["loo_deg"] for r in gb["runs"][cfg].values()]) for cfg in vals]
        print(f"  ground-truth centroid at the {band:4s} band lowers the mean angular error by "
              f"{min(loo):.2f} to {max(loo):.2f} deg")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default="paper_results", help="folder with the per-run result files")
    args = ap.parse_args()
    R = args.results
    t1 = load(R, "table1_results.json")
    table1(t1)
    section_4_1(t1)
    table2(load(R, "seed5_full.json"))
    table3(load(R, "phaseB_results.json"))
    table4(R)
    section_4_4(R)
    table6(t1, load(R, "deploy_allruns.json"))
    table7(t1, load(R, "prune5_results.json"))
    table8(load(R, "guidance_seeds.json"), load(R, "guidance_bands.json"))


if __name__ == "__main__":
    main()
