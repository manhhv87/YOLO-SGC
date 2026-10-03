# -*- coding: utf-8 -*-
"""make_fig_dataset.py - figure with dataset samples and their ground-truth labels.

Four images from two recording sessions, covering the range of illumination and vegetation
density, each drawn with its four distance-band boxes. The bands use shades of one hue whose
lightness changes steadily from near to far, so the order also reads in greyscale.

    python make_fig_dataset.py --list images.txt     # -> figures/dataset_samples.pdf

images.txt lists the four images, one path per line.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42          # embed text as TrueType, not outlines
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle
from PIL import Image

# near -> far, light -> dark (validated ordinal scale)
BANDS = [("near", "#00E0FF"), ("mid_near", "#FFEA00"),
         ("mid_far", "#FF2E93"), ("far", "#B96BFF")]
CLS = ["near", "mid_near", "mid_far", "far"]

ap = argparse.ArgumentParser()
ap.add_argument("--list", default="images.txt", help="file listing the four images, one path per line")
ap.add_argument("--out", default="figures/dataset_samples.pdf")
ap.add_argument("--lw", type=float, default=1.5, help="box line width")
args = ap.parse_args()


def load(p):
    lf = p.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
    a = np.loadtxt(lf).reshape(-1, 5)
    return np.asarray(Image.open(p).convert("RGB")), a[np.argsort(a[:, 0])]


def main():
    paths = [x.strip() for x in open(args.list) if x.strip()][:4]
    assert len(paths) == 4, f"four images required, got {len(paths)}"

    fig, axes = plt.subplots(1, 4, figsize=(11.0, 3.05))
    fig.subplots_adjust(left=0.004, right=0.996, top=0.985, bottom=0.145,
                        wspace=0.018)
    for k, (ax, p) in enumerate(zip(axes, paths)):
        img, lab = load(p)
        H, W = img.shape[:2]
        ax.imshow(img)
        for row in lab:
            c = int(row[0])
            cx, cy, w, h = row[1:] * np.array([W, H, W, H])
            ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h,
                                   fill=False, lw=args.lw, edgecolor=BANDS[c][1],
                                   path_effects=[pe.withStroke(linewidth=args.lw + 0.7,
                                                               foreground="#101010",
                                                               alpha=0.30)]))
        ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
        ax.text(0.022, 0.028, f"({'abcd'[k]})", transform=ax.transAxes,
                fontsize=10.5, color="white", va="bottom", ha="left",
                path_effects=[pe.withStroke(linewidth=2.4, foreground="#101010",
                                            alpha=0.75)])

    handles = [plt.Line2D([], [], color=c, lw=2.6, label=n) for n, c in BANDS]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=10.5, handlelength=1.7, columnspacing=2.4,
               handletextpad=0.6, bbox_to_anchor=(0.5, -0.006))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, format="pdf")
    print(f"-> {args.out}")
    for p in paths:
        print(f"   {os.path.basename(p)}")


if __name__ == "__main__":
    main()
