# -*- coding: utf-8 -*-
"""make_fig_scalemap.py — ScaleMap visualisation figure.

Four panels in a row:
  (a) input image;
  (b) scale map (160x160) on a [0,1] scale, min-max normalised per image,
      with an inset colour bar;
  (c) the map overlaid on the image with the predicted boxes of the four distance bands;
  (d) vertical profile: mean map value per image row, averaged over
      `--profile_n` test images (shaded band = +/-1 standard deviation across images).

Panel (d) is the quantitative evidence for a vertical-position prior: the map value rises
monotonically from the top of the image (far) to the bottom (near). Note that the mean
map value *inside the ground-truth boxes* is nearly equal across the four bands (~0.94),
because every box contains both soil and foliage, so that number is not used to annotate boxes.

    python make_fig_scalemap.py                       # -> figures_paper/fig_scalemap_vis.pdf
    python make_fig_scalemap.py --img <test image stem>

The --out extension selects the format (.pdf for the paper, .png for a quick look).
Run from the repository root.
"""
import argparse, glob, json, os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["pdf.compression"] = 9

ap = argparse.ArgumentParser()
ap.add_argument("--data_root", default="datasets")
ap.add_argument("--split", default="test")
ap.add_argument("--ckpt", default="weights/main/yolov8s_sgc_p345/best.pt")
ap.add_argument("--img", default="20250330_161021_000", help="stem of the test image to display")
ap.add_argument("--out", default="figures_paper/fig_scalemap_vis.pdf")
ap.add_argument("--profile_n", type=int, default=150, help="number of images used for the profile in panel (d)")
ap.add_argument("--profile_cache", default="paper_results/scalemap_profile.json")
ap.add_argument("--cmap", default="turbo")
ap.add_argument("--dpi", type=int, default=200)
args = ap.parse_args()

PANEL, GAP = 640, 12
FS_LETTER, FS_AX, FS_TICK, FS_BOX = 18.0, 15.0, 13.0, 15.0
CACHE_VERSION = 2
C_ALL, C_SOIL, C_VEG = "#1F4E79", "#C8791E", "#2E8B57"


def stroke(lw=2.6, fg="#1a1a1a"):
    return [pe.Stroke(linewidth=lw, foreground=fg), pe.Normal()]


def hook_scalemap(model):
    """Hook the ScaleMapHead and return a dict holding its output at each forward pass."""
    cap = {}
    for _, m in model.model.model.named_modules():
        if m.__class__.__name__ == "ScaleMapHead":
            m.register_forward_hook(lambda mod, i, o: cap.__setitem__("map", o.detach().cpu()))
            return cap
    raise RuntimeError("ScaleMapHead not found in the checkpoint")


def norm01(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def scan_test_set(model, cap, n):
    """Pool over n test images:
      - vertical profile of the map (all pixels / soil pixels / vegetation pixels by ExG);
      - variance decomposition of the map on image row position and on Excess-Green.
    Results are cached to JSON so the paper quotes the same source as the figure.
    """
    if os.path.exists(args.profile_cache):
        d = json.load(open(args.profile_cache))
        if d.get("n") == n and d.get("version") == CACHE_VERSION:
            return d
    P = {"all": [], "soil": [], "veg": []}
    r2 = {"row": [], "exg": [], "both": []}
    cr = {"row": [], "exg": []}
    for imgp in sorted(glob.glob(f"{args.data_root}/{args.split}/images/*"))[:n]:
        model.predict(imgp, imgsz=640, verbose=False)
        x = norm01(cap["map"][0, 0].numpy().astype(np.float64))
        Hm, Wm = x.shape
        im = cv2.resize(cv2.cvtColor(cv2.imread(imgp), cv2.COLOR_BGR2RGB).astype(np.float64),
                        (Wm, Hm), interpolation=cv2.INTER_AREA)
        exg = (2 * im[..., 1] - im[..., 0] - im[..., 2]) / (im.sum(2) + 1e-6)
        row = np.repeat(np.linspace(0, 1, Hm)[:, None], Wm, axis=1)   # 0 = top of image (far)

        xf, rf, ef = x.ravel(), row.ravel(), exg.ravel()
        var = xf.var()

        def r2_of(*preds):
            A = np.column_stack([np.ones_like(xf)] + list(preds))
            beta, *_ = np.linalg.lstsq(A, xf, rcond=None)
            return float(1 - ((xf - A @ beta) ** 2).mean() / var)

        r2["row"].append(r2_of(rf))
        r2["exg"].append(r2_of(ef))
        r2["both"].append(r2_of(rf, ef))
        cr["row"].append(float(np.corrcoef(xf, rf)[0, 1]))
        cr["exg"].append(float(np.corrcoef(xf, ef)[0, 1]))

        lo, hi = np.percentile(ef, 30), np.percentile(ef, 70)
        P["all"].append(x.mean(1))
        for key, msk in (("soil", exg <= lo), ("veg", exg >= hi)):
            cnt = msk.sum(1)
            P[key].append(np.where(cnt > 3, (x * msk).sum(1) / np.maximum(cnt, 1), np.nan))

    d = {"version": CACHE_VERSION, "n": len(P["all"]),
         "sd_all": np.nanstd(np.stack(P["all"]), 0).tolist(),
         **{f"prof_{k}": np.nanmean(np.stack(v), 0).tolist() for k, v in P.items()},
         **{f"r2_{k}": float(np.mean(v)) for k, v in r2.items()},
         **{f"corr_{k}": float(np.mean(v)) for k, v in cr.items()}}
    d["r2_row_given_exg"] = d["r2_both"] - d["r2_exg"]
    d["r2_exg_given_row"] = d["r2_both"] - d["r2_row"]
    os.makedirs(os.path.dirname(args.profile_cache), exist_ok=True)
    json.dump(d, open(args.profile_cache, "w"), indent=1)
    return d


def deciles(p):
    p = np.asarray(p)
    return [float(np.nanmean(p[int(i / 10 * len(p)):int((i + 1) / 10 * len(p))])) for i in range(10)]


def panel(fig, i, fig_w, fig_h):
    return fig.add_axes([i * (PANEL + GAP) / fig_w, 0.0, PANEL / fig_w, PANEL / fig_h])


def letter(ax, i):
    ax.text(0.018, 0.982, f"({chr(97 + i)})", transform=ax.transAxes, ha="left", va="top",
            fontsize=FS_LETTER, fontweight="bold", color="#111111", zorder=8,
            bbox=dict(boxstyle="square,pad=0.35", facecolor="white", alpha=0.86, edgecolor="none"))


def bare(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#b0b0b0")
        sp.set_linewidth(0.8)


def main():
    from ultralytics import YOLO
    model = YOLO(args.ckpt)
    cap = hook_scalemap(model)

    imgp = glob.glob(f"{args.data_root}/{args.split}/images/{args.img}*")[0]
    res = model.predict(imgp, imgsz=640, verbose=False)[0]
    smap = cap["map"][0, 0].numpy()
    img = cv2.cvtColor(cv2.imread(imgp), cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    disp = norm01(cv2.resize(smap, (W, H), interpolation=cv2.INTER_LINEAR))

    st = scan_test_set(model, cap, args.profile_n)

    fig_w, fig_h = 4 * PANEL + 3 * GAP, PANEL
    fig = plt.figure(figsize=(fig_w / args.dpi, fig_h / args.dpi), dpi=args.dpi)
    fig.patch.set_facecolor("white")

    # (a) input image
    ax = panel(fig, 0, fig_w, fig_h)
    ax.imshow(img, interpolation="none")
    bare(ax)
    letter(ax, 0)

    # (b) scale map with inset colour bar
    ax = panel(fig, 1, fig_w, fig_h)
    im = ax.imshow(disp, cmap=args.cmap, vmin=0, vmax=1, interpolation="none")
    bare(ax)
    letter(ax, 1)
    cax = ax.inset_axes([0.32, 0.105, 0.60, 0.030])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0, 0.5, 1])
    cb.outline.set(edgecolor="white", linewidth=1.0)
    cax.tick_params(labelsize=FS_TICK, colors="white", length=3, width=1.0, pad=2)
    for t in cax.get_xticklabels():
        t.set_path_effects(stroke(2.2))
    cax.text(-0.03, 0.5, "scale map", transform=cax.transAxes, ha="right", va="center",
             fontsize=FS_TICK, color="white", path_effects=stroke(2.2))

    # (c) overlay with predicted boxes
    ax = panel(fig, 2, fig_w, fig_h)
    ax.imshow(img, interpolation="none")
    ax.imshow(disp, cmap=args.cmap, alpha=0.45, vmin=0, vmax=1, interpolation="none")
    bare(ax)
    letter(ax, 2)
    boxes, cls, conf = (res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy().astype(int),
                        res.boxes.conf.cpu().numpy())
    for c in range(4):
        idx = np.where(cls == c)[0]
        if not len(idx):
            continue
        x1, y1, x2, y2 = boxes[idx[np.argmax(conf[idx])]]
        r = Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="white", lw=1.8, zorder=5)
        r.set_path_effects(stroke(3.4))
        ax.add_patch(r)
        ax.text(x1 + 6, y1 + 6, res.names[c], ha="left", va="top", fontsize=FS_BOX, color="#111111",
                zorder=6, bbox=dict(boxstyle="square,pad=0.25", facecolor="white", alpha=0.86,
                                    edgecolor="none"))

    # (d) vertical profile of the map, pooled over the test set
    box = panel(fig, 3, fig_w, fig_h)
    box.set_axis_off()
    ax = box.inset_axes([0.300, 0.225, 0.620, 0.715])
    prof_m, prof_sd = np.array(st["prof_all"]), np.array(st["sd_all"])
    y = np.linspace(0, 1, len(prof_m))
    ax.fill_betweenx(y, prof_m - prof_sd, prof_m + prof_sd, color=C_ALL, alpha=0.16, lw=0)
    ax.plot(np.array(st["prof_soil"]), y, color=C_SOIL, lw=1.8, ls=(0, (4.5, 2.2)), label="soil px")
    ax.plot(np.array(st["prof_veg"]), y, color=C_VEG, lw=1.8, ls=(0, (1, 1.7)), label="crop px")
    ax.plot(prof_m, y, color=C_ALL, lw=2.0, label="all px")
    ax.set_ylim(1, 0)                      # row 0 (top of image = far) at the top
    ax.set_xticks([0.7, 0.8, 0.9, 1.0])
    ax.legend(fontsize=FS_TICK - 1, loc="center left", frameon=True, framealpha=0.85,
              edgecolor="none", handlelength=1.9, borderpad=0.3, labelspacing=0.25,
              handletextpad=0.5)
    ax.set_xlabel("Scale-map value", fontsize=FS_AX, labelpad=2)
    ax.set_ylabel("Image row", fontsize=FS_AX, labelpad=2)
    ax.tick_params(labelsize=FS_TICK, pad=2)
    ax.grid(alpha=0.25, lw=0.7)
    for sp in ax.spines.values():
        sp.set_color("#888888")
    ax.text(0.04, 0.945, "far field", transform=ax.transAxes, va="top",
            fontsize=FS_TICK, color="#444444")
    ax.text(0.04, 0.03, "near field", transform=ax.transAxes, va="bottom",
            fontsize=FS_TICK, color="#444444")
    box.text(0.5, 0.012, f"$n={st['n']}$ test images", transform=box.transAxes, ha="center",
             va="bottom", fontsize=FS_TICK - 1, color="#444444")
    letter(box, 3)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, facecolor="white")
    print(f"-> {args.out}  ({fig_w / args.dpi:.2f}x{fig_h / args.dpi:.2f} in, "
          f"{os.path.getsize(args.out) / 1e6:.2f} MB)")
    print(f"   image shown: {os.path.basename(imgp)}\n")
    print(f"   SỐ TRÍCH VÀO BÀI (n={st['n']}, cache {args.profile_cache}):")
    print(f"     R^2 row = {st['r2_row']:.3f} (r={st['corr_row']:+.3f}) | "
          f"R^2 ExG = {st['r2_exg']:.3f} (r={st['corr_exg']:+.3f}) | R^2 both = {st['r2_both']:.3f}")
    print(f"     row given ExG = {st['r2_row_given_exg']:.3f} | "
          f"ExG given row = {st['r2_exg_given_row']:.3f}")
    for k, lab in (("all", "all "), ("soil", "soil"), ("veg", "veg ")):
        d = deciles(st[f"prof_{k}"])
        print(f"     deciles {lab}: {d[0]:.2f} -> {d[-1]:.2f}  "
              f"(monotone increasing: {all(d[i] < d[i+1] for i in range(9))})")


if __name__ == "__main__":
    main()
