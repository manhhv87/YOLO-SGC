# -*- coding: utf-8 -*-
"""make_fig_guidance.py — guidance-line figure.

Four representative test images, each with three lines: ground truth (through the four
box centroids), the classical Excess-Green detector, and the YOLOv8s-SGC-P345 pipeline.
A shared legend sits just above the panels; the LaTeX caption serves as the title.

    python make_fig_guidance.py --layout 1x4    # one row of four panels, used in the paper
    python make_fig_guidance.py                 # 2x2 grid
    python make_fig_guidance.py --title         # add a title above the panels

The --out extension selects the format: .pdf (vector, used in the paper; lines and text
are vector, the background image is embedded at 640x640) or .png for a quick look.

Run from the repository root.
"""
import argparse, glob, os
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

matplotlib.rcParams["pdf.fonttype"] = 42   # TrueType, avoids Type 3 which many journals reject
matplotlib.rcParams["pdf.compression"] = 9

ap = argparse.ArgumentParser()
ap.add_argument("--data_root", default="datasets")
ap.add_argument("--split", default="test")
ap.add_argument("--sgc", default="weights/main/yolov8s_sgc_p345/best.pt")
ap.add_argument("--device", default="0")
ap.add_argument("--out", default="figures_paper/guidance_line.pdf")
ap.add_argument("--layout", default="2x2", choices=["2x2", "1x4"])
ap.add_argument("--title", action="store_true", help="add a title above the panels")
ap.add_argument("--dpi", type=int, default=200)
args = ap.parse_args()

# The four representative images
STEMS = ["image_0380_jpg.rf.9a407e5ab6e65d506f05bec3795796ea",
         "20250330_161021_090_jpg.rf.b224d7a3f1c771a5bf4f5f8d0e3ac443",
         "image_0979_jpg.rf.60baae1cfec2ad7bc740f84a59271e5b",
         "20250330_161021_177_jpg.rf.a94f10232b510668a71bfe98a8f09a63"]

C_GT, C_CLS, C_SGC = "#00D2FF", "#E8352B", "#1FBF4F"
# Dash and dot periods are in points and matplotlib scales them by the line width.
LS_GT, LS_CLS, LS_SGC = "-", (0, (3.8, 2.6)), (0, (1, 1.8))


def stroke(lw):
    return [pe.Stroke(linewidth=lw, foreground="#1a1a1a", alpha=0.5), pe.Normal()]

PANEL = 640    # panel side in px; source images are 640x640
GAP = 12       # gap between panels in px

# Font sizes follow the printed width: the 2x2 layout prints at ~0.86\linewidth (~5.6 in),
# the 1x4 layout at \linewidth (6.5 in), so text is enlarged to offset the reduction.
STYLE = {"2x2": dict(rows=2, cols=2, legend_h=62, title_h=52, fs_leg=10.5, fs_letter=11.5,
                     fs_metric=10.0, fs_title=12.5, lw=(2.4, 1.6, 1.5), lw_leg=(1.8, 1.8, 2.0),
                     dot=14, hlen=2.2, cspace=1.6),
         "1x4": dict(rows=1, cols=4, legend_h=88, title_h=66, fs_leg=18.0, fs_letter=18.0,
                     fs_metric=17.0, fs_title=21.0, lw=(3.2, 2.0, 1.8), lw_leg=(2.2, 2.2, 2.4),
                     dot=24, hlen=2.8, cspace=2.6)}


# ---------------- guidance-line geometry (matches eval_guidance_line.py) ----------------
def fit_line(cx, cy):
    cx, cy = np.asarray(cx, float), np.asarray(cy, float)
    if len(cx) < 2:
        return None
    return np.polyfit(cy, cx, 1)


def ang_deg(m):
    return np.degrees(np.arctan(m))


def classical_line(rgb, n_bands=24):
    H, W, _ = rgb.shape
    r, g, b = rgb[..., 0].astype(float), rgb[..., 1].astype(float), rgb[..., 2].astype(float)
    s = r + g + b + 1e-6
    exg = 2 * g / s - r / s - b / s
    veg = (exg > max(0.02, np.percentile(exg, 55))).astype(float)
    lo0, hi0 = int(0.38 * W), int(0.62 * W)
    seed = veg[int(0.8 * H):, lo0:hi0].mean(0)
    ks = np.convolve(seed, np.ones(max(3, seed.size // 8)) / max(3, seed.size // 8), mode="same")
    cx_prev = (lo0 + int(np.argmin(ks))) / W
    win = 0.11
    xs, ys = [], []
    for k in range(n_bands - 1, -1, -1):
        y0, y1 = int(k / n_bands * H), int((k + 1) / n_bands * H)
        lo, hi = int(max(0, cx_prev - win) * W), int(min(1, cx_prev + win) * W)
        if hi - lo < 5:
            lo, hi = int(0.35 * W), int(0.65 * W)
        col = veg[y0:y1, lo:hi].mean(0)
        w = max(3, int(0.12 * col.size))
        sm = np.convolve(col, np.ones(w) / w, mode="same")
        pos = (np.arange(len(sm)) + lo) / W
        score = sm + 0.6 * np.abs(pos - cx_prev)
        j = int(np.argmin(score))
        cx = (lo + j) / W
        xs.append(cx)
        ys.append((y0 + y1) / 2 / H)
        cx_prev = 0.6 * cx + 0.4 * cx_prev
    xs, ys = np.array(xs), np.array(ys)
    m, b = np.polyfit(ys, xs, 1)
    res = np.abs(xs - (m * ys + b))
    keep = res < np.percentile(res, 75)
    if keep.sum() >= 4:
        m, b = np.polyfit(ys[keep], xs[keep], 1)
    return m, b


def yolo_line(model, imgp, device):
    r = model(imgp, device=device, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return None
    xywhn = r.boxes.xywhn.cpu().numpy()
    cls = r.boxes.cls.cpu().numpy().astype(int)
    conf = r.boxes.conf.cpu().numpy()
    cx, cy = [], []
    for c in range(4):
        idx = np.where(cls == c)[0]
        if len(idx) == 0:
            continue
        best = idx[np.argmax(conf[idx])]
        cx.append(xywhn[best, 0])
        cy.append(xywhn[best, 1])
    return fit_line(cx, cy) if len(cx) >= 2 else None


def draw_line(ax, m, b, y0, color, ls, lw, size, z=3):
    """x = m*y + b in normalised coordinates, drawn in pixels."""
    yy = np.array([y0, 1.0])
    ln, = ax.plot((m * yy + b) * size, yy * size, color=color, linestyle=ls, lw=lw,
                  solid_capstyle="round", dash_capstyle="round", zorder=z)
    ln.set_path_effects(stroke(lw + 1.2))
    return ln


def main():
    from ultralytics import YOLO
    model = YOLO(args.sgc)

    st = STYLE[args.layout]
    rows, cols = st["rows"], st["cols"]
    title_h = st["title_h"] if args.title else 0
    fig_w = cols * PANEL + (cols - 1) * GAP
    fig_h = rows * PANEL + (rows - 1) * GAP + st["legend_h"] + title_h
    fig = plt.figure(figsize=(fig_w / args.dpi, fig_h / args.dpi), dpi=args.dpi)
    fig.patch.set_facecolor("white")

    for i, stem in enumerate(STEMS):
        imgp = glob.glob(f"{args.data_root}/{args.split}/images/{stem}*")[0]
        lf = f"{args.data_root}/{args.split}/labels/{stem}.txt"
        a = np.loadtxt(lf).reshape(-1, 5)
        a = a[np.argsort(a[:, 0])]                      # near -> far theo class id
        gm, gb = fit_line(a[:, 1], a[:, 2])
        rgb = np.array(Image.open(imgp).convert("RGB"))
        size = rgb.shape[0]
        cm, cb = classical_line(rgb)
        pm, pb = yolo_line(model, imgp, args.device)

        r, c = divmod(i, cols)
        ax = fig.add_axes([c * (PANEL + GAP) / fig_w,
                           (rows - 1 - r) * (PANEL + GAP) / fig_h,
                           PANEL / fig_w, PANEL / fig_h])
        ax.imshow(rgb, interpolation="none")   # embed the 640x640 image as is, no resampling
        ax.set_xlim(0, size)
        ax.set_ylim(size, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#b0b0b0")
            sp.set_linewidth(0.8)

        y0 = max(0.0, float(a[:, 2].min()) - 0.03)      # from the farthest band down to the bottom edge
        w_gt, w_cls, w_sgc = st["lw"]
        draw_line(ax, gm, gb, y0, C_GT, LS_GT, w_gt, size, z=3)
        draw_line(ax, cm, cb, y0, C_CLS, LS_CLS, w_cls, size, z=5)
        draw_line(ax, pm, pb, y0, C_SGC, LS_SGC, w_sgc, size, z=5)
        ax.scatter(a[:, 1] * size, a[:, 2] * size, s=st["dot"], facecolor=C_GT,
                   edgecolor="#1a1a1a", linewidths=1.0, zorder=4)

        # panel label and the angular error of this image
        box = dict(boxstyle="square,pad=0.35", facecolor="white", alpha=0.86, edgecolor="none")
        ax.text(0.018, 0.982, f"({chr(97 + i)})", transform=ax.transAxes, ha="left", va="top",
                fontsize=st["fs_letter"], fontweight="bold", color="#111111", zorder=6, bbox=box)
        txt = (f"ExG  {abs(ang_deg(cm) - ang_deg(gm)):.1f}°\n"
               f"SGC  {abs(ang_deg(pm) - ang_deg(gm)):.1f}°")
        ax.text(0.018, 0.018, txt, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=st["fs_metric"], linespacing=1.3, color="#111111", zorder=6, bbox=box)

    lg, lc, ls_ = st["lw_leg"]
    handles = [Line2D([], [], color=C_GT, ls=LS_GT, lw=lg, path_effects=stroke(lg + 1.2)),
               Line2D([], [], color=C_CLS, ls=LS_CLS, lw=lc, path_effects=stroke(lc + 1.2)),
               Line2D([], [], color=C_SGC, ls=LS_SGC, lw=ls_, path_effects=stroke(ls_ + 1.2))]
    labels = ["Ground truth", "Classical ExG detector", "YOLOv8s-SGC-P345 (ours)"]
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 1.0 - title_h / fig_h), ncol=3,
               frameon=False, fontsize=st["fs_leg"], handlelength=st["hlen"],
               columnspacing=st["cspace"], handletextpad=0.8, borderpad=0.1)
    if args.title:
        fig.text(0.5, 1.0 - 8 / fig_h,
                 "Guidance line: GT vs classical crop-row detector (ExG) vs YOLOv8s-SGC-P345",
                 ha="center", va="top", fontsize=st["fs_title"])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, facecolor="white")
    print(f"-> {args.out}  ({fig_w / args.dpi:.2f}x{fig_h / args.dpi:.2f} in, "
          f"{os.path.getsize(args.out) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
