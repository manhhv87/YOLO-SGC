# -*- coding: utf-8 -*-
"""eval_guidance_bands.py - guidance-line error by distance band, five seeds per configuration.

Uses the definitions of Table 8 (eval_guidance_line.py, eval_guidance_seeds.py): confidence threshold
0.25, the line x = m*y + b through the box centroids, and the angular error against the ground-truth
line through the four labelled centroids. The highest-confidence box of each class comes from
paper_results/guidance_cache/, one JSON file per checkpoint; a missing file is rebuilt from the
checkpoint (prediction at conf 0.001, NMS IoU 0.7), so the analysis can be rerun from the cache alone.

For each checkpoint and each band (near, mid_near, mid_far, far):
  present  % of images with a box of the band (conf >= 0.25)
  dx_px    mean |x of predicted centroid - x of labelled centroid| in pixels of the 640 image,
           over the images with a box of the band
  lowiou   % of band boxes with IoU < 0.5 to the labelled box of the same band
  loo_deg  mean reduction of the angular error when the predicted centroid of the band is replaced
           by the labelled one (images with a line and a box of the band)

    python eval_guidance_bands.py        # -> paper_results/guidance_bands.json
"""
import argparse, glob, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)             # the Ultralytics fork of this repository
ap = argparse.ArgumentParser()
ap.add_argument("--device", default="0")
args = ap.parse_args()
OUT = "paper_results/guidance_bands.json"
CACHE = "paper_results/guidance_cache"
TAU = 0.25
BANDS = ["near", "mid_near", "mid_far", "far"]
CFGS = ["yolov8s", "yolov8s_sgc_p345", "yolov8s_sgc_p45"]


def ckpts(c):
    """The five checkpoints (seeds 0-4), as in eval_guidance_seeds.py."""
    return ([f"weights/main/{c}/best.pt"]
            + [f"weights/seed/{c}_s{s}/best.pt" for s in (1, 2)]
            + [f"runs/seed5/{c}_s{s}/weights/best.pt" for s in (3, 4)])


def fit_line(cx, cy):
    return np.polyfit(np.asarray(cy, float), np.asarray(cx, float), 1)   # x = m*y + b


def ang_deg(m):
    return np.degrees(np.arctan(m))


def iou_xywh(p, g):
    ix = max(0.0, min(p[0] + p[2] / 2, g[0] + g[2] / 2) - max(p[0] - p[2] / 2, g[0] - g[2] / 2))
    iy = max(0.0, min(p[1] + p[3] / 2, g[1] + g[3] / 2) - max(p[1] - p[3] / 2, g[1] - g[3] / 2))
    inter = ix * iy
    return inter / (p[2] * p[3] + g[2] * g[3] - inter)


def load_items():
    """Labels of the test images with at least two boxes, sorted by class, in label-file order."""
    out = []
    for lf in sorted(glob.glob("datasets/test/labels/*.txt")):
        stem = os.path.basename(lf)[:-4]
        if not glob.glob(f"datasets/test/images/{stem}*"):
            continue
        a = np.loadtxt(lf).reshape(-1, 5)
        if len(a) >= 2:
            out.append(a[np.argsort(a[:, 0])])
    return out


def load_images():
    """Image paths in the same order as load_items()."""
    out = []
    for lf in sorted(glob.glob("datasets/test/labels/*.txt")):
        stem = os.path.basename(lf)[:-4]
        ims = glob.glob(f"datasets/test/images/{stem}*")
        if not ims:
            continue
        if len(np.loadtxt(lf).reshape(-1, 5)) >= 2:
            out.append(ims[0])
    return out


def top_boxes(pt, images):
    """Highest-confidence box of each class on every image: {class: [conf, cx, cy, w, h]}."""
    from ultralytics import YOLO
    model = YOLO(pt)
    rows = []
    for imgp in images:
        r = model.predict(imgp, conf=0.001, iou=0.7, device=args.device, verbose=False)[0]
        top = {}
        if r.boxes is not None and len(r.boxes):
            xywhn = r.boxes.xywhn.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            conf = r.boxes.conf.cpu().numpy()
            for c in range(4):
                idx = np.where(cls == c)[0]
                if len(idx):
                    j = idx[np.argmax(conf[idx])]
                    top[str(c)] = [float(conf[j])] + [float(v) for v in xywhn[j]]
        rows.append(top)
    return rows


def one(rows, items):
    pres = {b: [] for b in range(4)}
    dx = {b: [] for b in range(4)}
    low = {b: [] for b in range(4)}
    loo = {b: [] for b in range(4)}
    ang = []
    for top, a in zip(rows, items):
        sel = {int(k): v for k, v in top.items() if v[0] >= TAU}
        g = {int(r[0]): r for r in a}
        for b in range(4):
            pres[b].append(b in sel)
            if b in sel and b in g:
                dx[b].append(abs(sel[b][1] - g[b][1]) * 640)
                low[b].append(iou_xywh(sel[b][1:], g[b][1:]) < 0.5)
        if len(sel) < 2:
            continue
        gm, _ = fit_line(a[:, 1], a[:, 2])
        ks = sorted(sel)
        pm, _ = fit_line([sel[k][1] for k in ks], [sel[k][2] for k in ks])
        err = abs(ang_deg(pm) - ang_deg(gm))
        ang.append(err)
        for b in range(4):
            if b in sel and b in g:
                s2 = dict(sel)
                s2[b] = [1.0] + list(g[b][1:])
                pm2, _ = fit_line([s2[k][1] for k in ks], [s2[k][2] for k in ks])
                loo[b].append(err - abs(ang_deg(pm2) - ang_deg(gm)))
    return {"ang_mean": float(np.mean(ang)), "n_lines": len(ang),
            "bands": {BANDS[b]: {"present_pct": 100 * float(np.mean(pres[b])), "dx_px": float(np.mean(dx[b])),
                                 "lowiou_pct": 100 * float(np.mean(low[b])), "loo_deg": float(np.mean(loo[b]))}
                      for b in range(4)}}


def main():
    items = load_items()
    images = None
    res = {"tau": TAU, "n_images": len(items), "image_px": 640,
           "definitions": {"present_pct": "% images with a box of the band at conf >= tau",
                           "dx_px": "mean |x_pred - x_gt| of the band centroid, pixels",
                           "lowiou_pct": "% predicted band boxes with IoU < 0.5 to the ground-truth box of the band",
                           "loo_deg": "mean reduction of the angular error when the predicted centroid of the band is "
                                      "replaced by the ground-truth centroid (images with a line and a box of the band)"},
           "runs": {}}
    for c in CFGS:
        res["runs"][c] = {}
        for pt in ckpts(c):
            path = os.path.join(CACHE, pt.replace("/", "__") + ".json")
            if not os.path.exists(path):
                images = images or load_images()
                os.makedirs(CACHE, exist_ok=True)
                json.dump(top_boxes(pt, images), open(path, "w"))
            rows = json.load(open(path))
            assert len(rows) == len(items)
            r = one(rows, items)
            res["runs"][c][pt] = r
            b = r["bands"]
            print(f"{c:18s} {pt.split('/')[-3] if 'runs/' in pt else pt.split('/')[-2]:22s} ang {r['ang_mean']:.3f}  "
                  + "  ".join(f"{k}: dx {b[k]['dx_px']:.1f} low {b[k]['lowiou_pct']:.1f} loo {b[k]['loo_deg']:+.2f}"
                              for k in ("near", "far")), flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
