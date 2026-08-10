# -*- coding: utf-8 -*-
"""eval_guidance_line.py — compare guidance-line quality across pipelines.

Each corn image shows one lane ahead of the robot with four boxes (near..far); the box
centroids define the guidance line. The ground-truth line is the fit x=m*y+b through the
four ground-truth centroids. Three methods are compared on the same test set:
  (1) Classical  — classical crop-row detector (Excess-Green plus soil-corridor tracking,
                   no learning); (2) YOLOv8s baseline; (3) YOLOv8s-SGC-P345, proposed.
Metrics follow the crop-row convention of Gong et al. (2024): angular error in degrees
against ground truth; lateral offset at the far band, normalised by image width; and the

rate at which a line is recovered.

Usage: python eval_guidance_line.py --split test --device 0
"""
import argparse, glob, os, numpy as np
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--data_root", default="datasets")
ap.add_argument("--split", default="test")
ap.add_argument("--device", default="0")
ap.add_argument("--sgc", default="weights/main/yolov8s_sgc_p345/best.pt")
ap.add_argument("--base", default="weights/main/yolov8s/best.pt")
ap.add_argument("--out", default="paper_results/guidance_line.md")
ap.add_argument("--viz", default="figures_paper/guidance_line.png")
args = ap.parse_args()


def fit_line(cx, cy):
    """x = m*y + b for a near-vertical line. Return (m,b) or None."""
    cx, cy = np.asarray(cx, float), np.asarray(cy, float)
    if len(cx) < 2: return None
    m, b = np.polyfit(cy, cx, 1)
    return m, b


def ang_deg(m):
    return np.degrees(np.arctan(m))  # angle against the image vertical


# ---------------- (1) Classical detector: ExG + soil-corridor tracking, centre prior ----------------
def classical_line(rgb, n_bands=24):
    H, W, _ = rgb.shape
    r, g, b = rgb[..., 0].astype(float), rgb[..., 1].astype(float), rgb[..., 2].astype(float)
    s = r + g + b + 1e-6
    exg = 2 * g / s - r / s - b / s
    veg = (exg > max(0.02, np.percentile(exg, 55))).astype(float)
    # initialise from the three bottom bands, find the vegetation minimum in the tight
    lo0, hi0 = int(0.38 * W), int(0.62 * W)
    seed = veg[int(0.8 * H):, lo0:hi0].mean(0)
    ks = np.convolve(seed, np.ones(max(3, seed.size // 8)) / max(3, seed.size // 8), mode="same")
    cx_prev = (lo0 + int(np.argmin(ks))) / W
    win = 0.11
    xs, ys = [], []
    for k in range(n_bands - 1, -1, -1):
        y0, y1 = int(k / n_bands * H), int((k + 1) / n_bands * H)
        lo, hi = int(max(0, cx_prev - win) * W), int(min(1, cx_prev + win) * W)
        if hi - lo < 5: lo, hi = int(0.35 * W), int(0.65 * W)
        col = veg[y0:y1, lo:hi].mean(0)
        w = max(3, int(0.12 * col.size))
        sm = np.convolve(col, np.ones(w) / w, mode="same")
        # continuity penalty: pick the column with least vegetation AND nearest cx_prev
        pos = (np.arange(len(sm)) + lo) / W
        score = sm + 0.6 * np.abs(pos - cx_prev)
        j = int(np.argmin(score))
        cx = (lo + j) / W
        xs.append(cx); ys.append((y0 + y1) / 2 / H)
        cx_prev = 0.6 * cx + 0.4 * cx_prev
    xs, ys = np.array(xs), np.array(ys)
    m, b = np.polyfit(ys, xs, 1)
    res = np.abs(xs - (m * ys + b)); keep = res < np.percentile(res, 75)
    if keep.sum() >= 4: m, b = np.polyfit(ys[keep], xs[keep], 1)
    return m, b


# ---------------- (2,3) YOLO pipeline: highest-confidence box per class -> centroid -> line ----------------
def yolo_line(model, imgp, device):
    r = model(imgp, device=device, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0: return None, None
    xywhn = r.boxes.xywhn.cpu().numpy()   # normalised cx,cy,w,h
    cls = r.boxes.cls.cpu().numpy().astype(int)
    conf = r.boxes.conf.cpu().numpy()
    cx, cy = [], []
    for c in range(4):
        idx = np.where(cls == c)[0]
        if len(idx) == 0: continue
        best = idx[np.argmax(conf[idx])]
        cx.append(xywhn[best, 0]); cy.append(xywhn[best, 1])
    if len(cx) < 2: return None, len(cx)
    return fit_line(cx, cy), len(cx)


def main():
    from ultralytics import YOLO
    lbls = sorted(glob.glob(f"{args.data_root}/{args.split}/labels/*.txt"))
    sgc = YOLO(args.sgc); base = YOLO(args.base)
    rows = {"Classical (ExG)": {"ang": [], "far": [], "ok": 0},
            "YOLOv8s": {"ang": [], "far": [], "ok": 0, "bands": []},
            "YOLOv8s-SGC-P345": {"ang": [], "far": [], "ok": 0, "bands": []}}
    N = 0
    for lf in lbls:
        stem = os.path.basename(lf)[:-4]
        ims = glob.glob(f"{args.data_root}/{args.split}/images/{stem}*")
        if not ims: continue
        imgp = ims[0]
        a = np.loadtxt(lf).reshape(-1, 5)
        if len(a) < 2: continue
        a = a[np.argsort(a[:, 0])]
        gm, gb = fit_line(a[:, 1], a[:, 2])
        far_y = a[a[:, 0].argmax(), 2]        # y of the farthest box
        gt_far_x = gm * far_y + gb
        N += 1
        rgb = np.array(Image.open(imgp).convert("RGB"))
        # (1) classical
        try:
            cm, cb = classical_line(rgb)
            rows["Classical (ExG)"]["ang"].append(abs(ang_deg(cm) - ang_deg(gm)))
            rows["Classical (ExG)"]["far"].append(abs((cm * far_y + cb) - gt_far_x))
            rows["Classical (ExG)"]["ok"] += 1
        except Exception:
            pass
        # (2,3) yolo pipelines
        for name, model in [("YOLOv8s", base), ("YOLOv8s-SGC-P345", sgc)]:
            res, nb = yolo_line(model, imgp, args.device)
            if nb is not None: rows[name]["bands"].append(nb)
            if res is None: continue
            pm, pb = res
            rows[name]["ang"].append(abs(ang_deg(pm) - ang_deg(gm)))
            rows[name]["far"].append(abs((pm * far_y + pb) - gt_far_x))
            rows[name]["ok"] += 1

    def stat(v):
        v = np.array(v); return v.mean(), np.median(v), v.std()
    lines = ["# Guidance-line comparison (F-05 / F-06)  —  test set, N=%d images" % N,
             "",
             "GT line = least-squares fit through the 4 GT box centroids (near→far). "
             "Angular error = |angle(pred) − angle(GT)| in degrees (angle w.r.t. image vertical). "
             "Far offset = |x_pred − x_GT| at the far-band row, normalized by image width. "
             "Line rate = fraction of images where a valid guidance line was extracted.", "",
             "| Method | Angular err (°) mean / median | Far offset (norm.) mean | Line rate |",
             "|---|---|---|---|"]
    for name, d in rows.items():
        if not d["ang"]:
            lines.append(f"| {name} | — | — | 0 |"); continue
        am, amed, asd = stat(d["ang"]); fm, fmed, fsd = stat(d["far"])
        lines.append(f"| {name} | {am:.2f} / {amed:.2f} | {fm:.3f} | {d['ok']/N:.2f} |")
    # completeness: in how many images the detector recovers all four bands
    for name in ("YOLOv8s", "YOLOv8s-SGC-P345"):
        bb = np.array(rows[name]["bands"]);
        if bb.size:
            lines.append("")
            lines.append(f"- {name}: all four bands in {100*(bb==4).mean():.1f}% of images (mean {bb.mean():.2f} bands/image)")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w").write("\n".join(lines))
    print("\n".join(lines))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
