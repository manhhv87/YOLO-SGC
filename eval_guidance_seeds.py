# -*- coding: utf-8 -*-
"""eval_guidance_seeds.py — guidance-line quality over five seeds for three pipelines.

Extends the single-model evaluation to mean +/- standard deviation over five seeds, with
a Welch t-test and Mann-Whitney test against the plain YOLOv8s baseline, the same
statistical protocol used for the detection tables. The classical ExG detector has no
learned parameters and is therefore run once.

    python eval_guidance_seeds.py --split test --device 0
"""
import argparse
import json
import os
import statistics as S
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--data_root", default="datasets")
ap.add_argument("--split", default="test")
ap.add_argument("--device", default="0")
ap.add_argument("--out", default="paper_results/guidance_seeds.json")
args = ap.parse_args()

# reuse the already validated helpers
import importlib.util
_spec = importlib.util.spec_from_file_location("_gl", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "eval_guidance_line.py"))
_argv, sys.argv = sys.argv, ["eval_guidance_line.py"]
_gl = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_gl)
sys.argv = _argv
fit_line, ang_deg, classical_line, yolo_line = (
    _gl.fit_line, _gl.ang_deg, _gl.classical_line, _gl.yolo_line)

CFG = [("yolov8s", "YOLOv8s"),
       ("yolov8s_sgc_p345", "YOLOv8s-SGC-P345"),
       ("yolov8s_sgc_p45", "YOLOv8s-SGC-P45")]


def ckpts(c):
    """Five checkpoint paths for seeds 0..4, same convention as table1_eval.py."""
    return ([f"weights/main/{c}/best.pt"]
            + [f"weights/seed/{c}_s{s}/best.pt" for s in (1, 2)]
            + [f"runs/seed5/{c}_s{s}/weights/best.pt" for s in (3, 4)])


def load_gt():
    """(image, ground-truth line coefficients, far-band y, ground-truth x there) per valid image."""
    import glob
    out = []
    for lf in sorted(glob.glob(f"{args.data_root}/{args.split}/labels/*.txt")):
        stem = os.path.basename(lf)[:-4]
        ims = glob.glob(f"{args.data_root}/{args.split}/images/{stem}*")
        if not ims:
            continue
        a = np.loadtxt(lf).reshape(-1, 5)
        if len(a) < 2:
            continue
        a = a[np.argsort(a[:, 0])]
        gm, gb = fit_line(a[:, 1], a[:, 2])
        far_y = a[a[:, 0].argmax(), 2]
        out.append((ims[0], gm, far_y, gm * far_y + gb))
    return out


def score(model, gts):
    ang, far, bands = [], [], []
    for imgp, gm, far_y, gt_far_x in gts:
        res, nb = yolo_line(model, imgp, args.device)
        if nb is not None:
            bands.append(nb)
        if res is None:
            continue
        pm, pb = res
        ang.append(abs(ang_deg(pm) - ang_deg(gm)))
        far.append(abs((pm * far_y + pb) - gt_far_x))
    b = np.array(bands)
    return {"ang_mean": float(np.mean(ang)), "ang_med": float(np.median(ang)),
            "far_mean": float(np.mean(far)), "four": float(100 * (b == 4).mean()),
            "bands_mean": float(b.mean())}


def stats(a, b):
    try:
        from scipy import stats as st
        return (float(st.ttest_ind(a, b, equal_var=False).pvalue),
                float(st.mannwhitneyu(a, b, alternative="two-sided").pvalue))
    except Exception:
        return float("nan"), float("nan")


def main():
    from ultralytics import YOLO
    gts = load_gt()
    print(f"{len(gts)} images with valid labels", flush=True)
    res = json.load(open(args.out)) if os.path.exists(args.out) else {}

    # the classical detector has no learned parameters, so it runs once
    if "classical" not in res:
        ang, far = [], []
        for imgp, gm, far_y, gt_far_x in gts:
            try:
                cm, cb = classical_line(np.array(Image.open(imgp).convert("RGB")))
            except Exception:
                continue
            ang.append(abs(ang_deg(cm) - ang_deg(gm)))
            far.append(abs((cm * far_y + cb) - gt_far_x))
        res["classical"] = {"ang_mean": float(np.mean(ang)), "ang_med": float(np.median(ang)),
                            "far_mean": float(np.mean(far))}
        print(f"Classical (ExG)  {res['classical']['ang_mean']:.2f} / "
              f"{res['classical']['ang_med']:.2f}", flush=True)
        json.dump(res, open(args.out, "w"), indent=1)

    for c, name in CFG:
        res.setdefault(c, {})
        for pt in ckpts(c):
            if pt in res[c]:
                continue
            if not os.path.exists(pt):
                print(f"THIẾU {pt}", flush=True); continue
            res[c][pt] = score(YOLO(pt), gts)
            r = res[c][pt]
            print(f"{name:20s} {os.path.basename(os.path.dirname(pt)):22s} "
                  f"ang={r['ang_mean']:.2f} 4-band={r['four']:.1f}%", flush=True)
            json.dump(res, open(args.out, "w"), indent=1)

    # report
    print(f"\n{'Method':20s} {'ang mean (deg)':>16s} {'ang median':>15s} "
          f"{'far offset':>15s} {'4-band (%)':>16s}")
    print("-" * 88)
    c0 = res["classical"]
    print(f"{'Classical (ExG)':20s} {c0['ang_mean']:16.2f} {c0['ang_med']:15.2f} "
          f"{c0['far_mean']:15.3f} {'n/a':>16s}")
    base = None
    tail = []
    for c, name in CFG:
        v = list(res[c].values())
        g = lambda k: [x[k] for x in v]
        ms = lambda k: (S.mean(g(k)), S.stdev(g(k)) if len(v) > 1 else 0.0)
        print(f"{name:20s} {ms('ang_mean')[0]:9.2f}±{ms('ang_mean')[1]:<5.2f} "
              f"{ms('ang_med')[0]:8.2f}±{ms('ang_med')[1]:<5.2f} "
              f"{ms('far_mean')[0]:8.3f}±{ms('far_mean')[1]:<5.3f} "
              f"{ms('four')[0]:9.1f}±{ms('four')[1]:<5.1f}  (n={len(v)})")
        if base is None:
            base = {k: g(k) for k in ("ang_mean", "four")}
        else:
            pa, pf = stats(g("ang_mean"), base["ang_mean"]), stats(g("four"), base["four"])
            tail.append(f"  {name:20s} ang d{S.mean(g('ang_mean'))-S.mean(base['ang_mean']):+.2f}deg "
                        f"p={pa[0]:.3f}/{pa[1]:.3f}   "
                        f"4-band d{S.mean(g('four'))-S.mean(base['four']):+.1f} points "
                        f"p={pf[0]:.3f}/{pf[1]:.3f}")
    print("\nAgainst YOLOv8s (Welch / Mann-Whitney):")
    for t in tail:
        print(t)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
