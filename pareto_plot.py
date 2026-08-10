# -*- coding: utf-8 -*-
"""
pareto_plot.py — accuracy-versus-parameters frontier (mAP@50 vs parameters) from eval_results.csv.

- Circle (o) = unpruned model, square (s) = pruned model.
- A dotted line joins each model to its pruned counterpart.
- Yellow star (*) = Pareto-optimal point (no model has fewer parameters and higher mAP).

Run once eval_results.csv exists:
    python pareto_plot.py --csv eval_results.csv --out pareto
Writes pareto.pdf and pareto.png
"""
import argparse, csv, re, math
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42   # TrueType, avoids Type 3
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="eval_results.csv")
ap.add_argument("--out", default="pareto")
ap.add_argument("--include-ablation", action="store_true", help="ve luon cac bien the ablation")
args = ap.parse_args()

COLORS = {"YOLOv5s": "#1f77b4", "YOLOv8s": "#ff7f0e", "YOLOv9s": "#2ca02c", "YOLO11s": "#d62728"}


def classify(name):
    """Suy ra (backbone, variant, pruned, ablation) tu ten thu muc run."""
    n = name.lower()
    pruned = "prune" in n
    ablation = any(k in n for k in ["amrf_fpn", "equalweight", "minimal", "noamrf", "noscale"])
    m = re.search(r"yolov?(\d+)", n)
    num = m.group(1) if m else "?"
    backbone = {"5": "YOLOv5s", "8": "YOLOv8s", "9": "YOLOv9s", "11": "YOLO11s"}.get(num, "YOLO?" + num)
    if "sgc" not in n:
        variant = "baseline"
    elif "p345" in n:
        variant = "P345"
    elif "p45" in n or "lite" in n:   # 'yolov8s_sgc_lite' chinh la P45
        variant = "P45"
    else:
        variant = "baseline"
    return backbone, variant, pruned, ablation


# --- read the CSV ---
rows = []
n_nan = 0
with open(args.csv, newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        try:
            params = float(r["Params(M)"]); mapv = float(r["mAP50"])
        except (KeyError, ValueError):
            continue
        if math.isnan(params) or math.isnan(mapv):   # cot Params bi nan -> khong ve duoc
            n_nan += 1
            continue
        bb, var, pr, ab = classify(r["model"])
        if ab and not args.include_ablation:
            continue
        label = bb if var == "baseline" else f"{bb}-SGC-{var}"
        rows.append(dict(bb=bb, var=var, pruned=pr, params=params, map=mapv, label=label))

if n_nan:
    print(f"CANH BAO: {n_nan} dong co Params(M)=nan -> da bo qua (hinh se thieu diem/trong). "
          f"Hay dien Params/GFLOPs (vd tu Results.xlsx) roi chay lai.")
if not rows:
    raise SystemExit("Khong co diem nao ve duoc (tat ca Params=nan?). Kiem tra cot Params(M) trong " + args.csv)

# --- Pareto frontier (skyline: it tham so nhat & mAP cao nhat) ---
sky = sorted(rows, key=lambda d: (d["params"], -d["map"]))
frontier, best = [], -1.0
for d in sky:
    if d["map"] > best + 1e-9:
        frontier.append(d); best = d["map"]

# --- plot ---
fig, axp = plt.subplots(figsize=(8, 6))

# dotted link between the unpruned and pruned model of the same (backbone, variant)
pair = {}
for d in rows:
    pair.setdefault((d["bb"], d["var"]), {})[d["pruned"]] = d
for (bb, var), pp in pair.items():
    if False in pp and True in pp:
        a, b = pp[False], pp[True]
        axp.plot([a["params"], b["params"]], [a["map"], b["map"]],
                 ls=":", color=COLORS.get(bb, "gray"), lw=1.0, alpha=0.7, zorder=1)

# points
for d in rows:
    axp.scatter(d["params"], d["map"], marker=("s" if d["pruned"] else "o"),
                s=70, facecolor=COLORS.get(d["bb"], "gray"), edgecolor="k", lw=0.6, zorder=3)

# frontier line, stars and labels
axp.plot([d["params"] for d in frontier], [d["map"] for d in frontier],
         ls="--", color="black", lw=1.2, zorder=2)
for d in frontier:
    axp.scatter(d["params"], d["map"], marker="*", s=340, facecolor="gold",
                edgecolor="k", lw=0.8, zorder=4)
    axp.annotate(d["label"] + (" (pruned)" if d["pruned"] else ""),
                 (d["params"], d["map"]), textcoords="offset points",
                 xytext=(9, -12), fontsize=8,
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec="none"))

# two legends: colour = backbone, marker = state
leg_bb = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markeredgecolor="k",
                 label=b, markersize=9) for b, c in COLORS.items()]
leg_mk = [Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markeredgecolor="k",
                 label="Unpruned", markersize=9),
          Line2D([0], [0], marker="s", color="w", markerfacecolor="gray", markeredgecolor="k",
                 label="Pruned", markersize=9),
          Line2D([0], [0], marker="*", color="w", markerfacecolor="gold", markeredgecolor="k",
                 label="Pareto-optimal", markersize=14)]
first = axp.legend(handles=leg_bb, loc="lower right", title="Backbone", fontsize=8)
axp.add_artist(first)
axp.legend(handles=leg_mk, loc="lower left", fontsize=8)

axp.set_xlabel("Parameters (M)")
axp.set_ylabel("mAP@50")
# No embedded title: the LaTeX caption serves that role.
axp.grid(True, ls=":", alpha=0.4)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"{args.out}.{ext}", dpi=300, bbox_inches="tight")

print("Cac model Pareto-optimal (frontier):")
for d in frontier:
    print(f"  {d['label']:24s} pruned={str(d['pruned']):5s} "
          f"params={d['params']:.2f}M  mAP50={d['map']:.4f}")
print(f"\nDa luu: {args.out}.pdf va {args.out}.png")
