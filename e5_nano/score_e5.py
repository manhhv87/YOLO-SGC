# -*- coding: utf-8 -*-
"""score_e5.py - scoring on a PC: accuracy of the TensorRT FP16 engines (predictions from the Nano) and trtexec timing.

    python e5_nano/score_e5.py --ref                        # FP32 reference: ONNX + onnxruntime CPU, same pipeline
    python e5_nano/score_e5.py                              # score results_nano/ -> paper_results/e5_trt.json

Predictions are matched to the labels and scored as in the Ultralytics DetectionValidator (8.3.160), so the
FP16 - FP32 difference of the same model at the same 640x640 input comes only from the engine arithmetic.
"""
import argparse, glob, json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(ROOT, "e5_nano")
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
os.environ["YOLO_AUTOINSTALL"] = "false"

import numpy as np

TEST = os.path.join(ROOT, "datasets/test")
FAR = 3                                   # names: 0 near, 1 mid_near, 2 mid_far, 3 far

# FP32 throughput (PyTorch, Nano) reported in the paper: Table 1 (unpruned) and Table 7 (pruned)
PAPER_FP32_FPS = {
    "yolov5s": 5.0, "yolov5s_sgc_p345": 4.5, "yolov5s_sgc_p45": 7.6,
    "yolov8s": 4.5, "yolov8s_sgc_p345": 5.2, "yolov8s_sgc_p45": 7.3,
    "yolov9s": 4.2, "yolov9s_sgc_p345": 4.6, "yolov9s_sgc_p45": 5.8,
    "yolov11s": 4.5, "yolov11s_sgc_p345": 5.2, "yolov11s_sgc_p45": 6.1,
    "yolov5s_pruned": 10.7, "yolov5s_sgc_p345_pruned": 7.2, "yolov5s_sgc_p45_pruned": 15.3,
    "yolov8s_pruned": 10.3, "yolov8s_sgc_p345_pruned": 9.3, "yolov8s_sgc_p45_pruned": 27.7,
    "yolov9s_pruned": 5.7, "yolov9s_sgc_p345_pruned": 6.3, "yolov9s_sgc_p45_pruned": 8.1,
    "yolov11s_pruned": 8.9, "yolov11s_sgc_p345_pruned": 9.8, "yolov11s_sgc_p45_pruned": 13.5,
}


def load_gt(names):
    """YOLO labels -> xyxy pixels, clipped to the image; duplicate rows removed as in verify_image_label."""
    import cv2
    gt = []
    for n in names:
        h, w = cv2.imread(os.path.join(TEST, "images", n)).shape[:2]
        f = os.path.join(TEST, "labels", os.path.splitext(n)[0] + ".txt")
        lb = np.loadtxt(f, ndmin=2, dtype=np.float32) if os.path.exists(f) and os.path.getsize(f) else np.zeros((0, 5), np.float32)
        assert lb.shape[1] == 5, "%s: labels are not 5-column boxes" % f
        if len(lb):
            _, i = np.unique(lb, axis=0, return_index=True)
            lb = lb[i]
        xyxy = np.stack([(lb[:, 1] - lb[:, 3] / 2) * w, (lb[:, 2] - lb[:, 4] / 2) * h,
                         (lb[:, 1] + lb[:, 3] / 2) * w, (lb[:, 2] + lb[:, 4] / 2) * h], 1)
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, w)
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, h)
        gt.append((lb[:, 0].astype(np.int64), xyxy))
    return gt


def match(pred_cls, true_cls, iou, iouv):
    """BaseValidator.match_predictions, branch without scipy."""
    correct = np.zeros((pred_cls.shape[0], iouv.shape[0]), dtype=bool)
    iou = iou * (true_cls[:, None] == pred_cls)
    for i, t in enumerate(iouv.tolist()):
        m = np.array(np.nonzero(iou >= t)).T
        if m.shape[0]:
            if m.shape[0] > 1:
                m = m[iou[m[:, 0], m[:, 1]].argsort()[::-1]]
                m = m[np.unique(m[:, 1], return_index=True)[1]]
                m = m[np.unique(m[:, 0], return_index=True)[1]]
            correct[m[:, 1].astype(int), i] = True
    return correct


def metrics(dets, img, names, gt):
    """dets (K,6) x1,y1,x2,y2,conf,cls in original-image coordinates; img (K,) image index. Returns mAP50, mAP50-95, far-class AP50."""
    import torch
    from ultralytics.utils.metrics import ap_per_class, box_iou
    iouv = np.linspace(0.5, 0.95, 10)
    tp, conf, pcls, tcls = [], [], [], []
    for k in range(len(names)):
        d = dets[img == k]
        gcls, gbox = gt[k]
        if len(gcls) and len(d):
            iou = box_iou(torch.from_numpy(gbox).float(), torch.from_numpy(d[:, :4]).float()).numpy()
            tp.append(match(d[:, 5].astype(np.int64), gcls, iou, iouv))
        else:
            tp.append(np.zeros((len(d), 10), bool))
        conf.append(d[:, 4])
        pcls.append(d[:, 5])
        tcls.append(gcls)
    tp, conf, pcls, tcls = [np.concatenate(v) for v in (tp, conf, pcls, tcls)]
    r = ap_per_class(tp, conf, pcls, tcls, plot=False)
    ap, ucls = r[5], r[6]                     # ap (nc,10), classes present in the labels
    far = float(ap[list(ucls).index(FAR), 0]) if FAR in ucls else float("nan")
    return {"map50": float(ap[:, 0].mean()), "map50_95": float(ap.mean()), "far_ap50": far,
            "n_det": int(len(dets))}


def ref_fp32(models):
    """FP32 predictions with onnxruntime on CPU, with the pre/post-processing of trt_common (as on the Nano)."""
    import cv2, onnxruntime as ort
    import trt_common as tc
    out = os.path.join(HERE, "ref_fp32")
    os.makedirs(out, exist_ok=True)
    names = sorted(os.listdir(os.path.join(TEST, "images")))
    for m in models:
        f = os.path.join(out, "pred_%s.npz" % m)
        if os.path.exists(f):
            continue
        sess = ort.InferenceSession(os.path.join(HERE, "onnx", m + ".onnx"), providers=["CPUExecutionProvider"])
        iname = sess.get_inputs()[0].name
        dets, idx = [], []
        for k, n in enumerate(names):
            bgr = cv2.imread(os.path.join(TEST, "images", n))
            x, r, pad = tc.preprocess(bgr)
            d = tc.postprocess(sess.run(None, {iname: x})[0], bgr.shape[:2], r, pad)
            dets.append(d)
            idx.append(np.full(len(d), k, np.int32))
        np.savez_compressed(f, dets=np.concatenate(dets), img=np.concatenate(idx), names=np.array(names))
        print("ref FP32  %-26s %d box" % (m, sum(len(d) for d in dets)), flush=True)


def load_pred(f):
    z = np.load(f)
    return z["dets"], z["img"], [str(s) for s in z["names"]]


def trtexec_timing(name, prec="fp16"):
    log = os.path.join(HERE, "results_nano", "%s_%s.log" % (name, prec))
    tj = os.path.join(HERE, "results_nano", "%s_%s_times.json" % (name, prec))
    if not os.path.exists(log):
        return None
    s = open(log, errors="ignore").read()
    out = {}
    g = re.findall(r"GPU Compute(?: Time)?: min = ([\d.]+) ms, max = ([\d.]+) ms, mean = ([\d.]+) ms, median = ([\d.]+) ms", s)
    if g:
        mn, mx, mean, med = map(float, g[-1])
        out.update(gpu_ms_min=mn, gpu_ms_max=mx, gpu_ms_mean=mean, gpu_ms_median=med, fps_gpu_mean=1000.0 / mean)
    h = re.findall(r"\[I\] Latency: min = ([\d.]+) ms, max = ([\d.]+) ms, mean = ([\d.]+) ms, median = ([\d.]+) ms, "
                   r"percentile\(99%\) = ([\d.]+) ms", s)
    if h:   # host latency = H2D + GPU compute + D2H of one query (trtexec)
        hmn, hmx, hmean, hmed, hp99 = map(float, h[-1])
        out.update(host_ms_mean=hmean, host_ms_median=hmed, host_ms_p99=hp99, fps_host_mean=1000.0 / hmean)
    q = re.findall(r"Throughput: ([\d.]+) qps", s)
    if q:
        out["throughput_qps"] = float(q[-1])
    v = re.findall(r"TensorRT version: ?([\d.]+)", s)
    if v:
        out["trt_version"] = v[-1]
    if os.path.exists(tj):
        t = json.load(open(tj))
        c = np.array([r.get("computeMs", r.get("latencyMs", np.nan)) for r in t], float)
        out.update(n_iter=int(len(c)), gpu_ms_p50=float(np.percentile(c, 50)),
                   gpu_ms_p95=float(np.percentile(c, 95)), gpu_ms_p99=float(np.percentile(c, 99)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", action="store_true", help="compute the FP32 reference for every ONNX model")
    ap.add_argument("--out", default=os.path.join(ROOT, "paper_results", "e5_trt.json"))
    args = ap.parse_args()
    os.chdir(ROOT)
    man = json.load(open(os.path.join(HERE, "onnx", "manifest.json")))
    if args.ref:
        ref_fp32(list(man))
        return

    names = sorted(os.listdir(os.path.join(TEST, "images")))
    gt = load_gt(names)
    res = {"note": "TensorRT engines built and timed with trtexec on the Jetson Nano; accuracy of the FP16 "
                   "engine vs the FP32 ONNX model under the same pre/post-processing (trt_common.py).",
           "device_state": {}, "models": {}}
    for k in ("start", "end"):
        f = os.path.join(HERE, "results_nano", "device_state_%s.txt" % k)
        if os.path.exists(f):
            res["device_state"][k] = open(f, errors="ignore").read()
    print("%-26s %7s %8s %6s | %7s %7s %6s | %7s %7s" % ("model", "FP32", "TRT16", "x", "mAP50", "FP16", "d",
                                                          "far", "d"))
    for m in man:
        e = {"source": man[m]["source"], "paper_fp32_fps": PAPER_FP32_FPS.get(m)}
        for prec in ("fp16", "fp32"):
            t = trtexec_timing(m, prec)
            if t:
                e["trt_%s" % prec] = t
        rf = os.path.join(HERE, "ref_fp32", "pred_%s.npz" % m)
        pf = os.path.join(HERE, "results_nano", "pred_%s_fp16.npz" % m)
        if os.path.exists(rf):
            d, i, nm = load_pred(rf)
            assert nm == names
            e["acc_onnx_fp32"] = metrics(d, i, names, gt)
        if os.path.exists(pf):
            d, i, nm = load_pred(pf)
            assert nm == names, "images on the Nano differ from the test set"
            e["acc_trt_fp16"] = metrics(d, i, names, gt)
            z = np.load(pf)
            tm = z["times_ms"]
            e["nano_eval_ms"] = {"read": float(tm[:, 0].mean()), "pre": float(tm[:, 1].mean()),
                                 "infer": float(tm[:, 2].mean()), "nms": float(tm[:, 3].mean())}
        res["models"][m] = e
        fp = e.get("trt_fp16", {}).get("fps_gpu_mean")
        a0, a1 = e.get("acc_onnx_fp32"), e.get("acc_trt_fp16")
        f = lambda v: "%7.4f" % v if v is not None else "      -"
        print("%-26s %7s %8s %6s | %s %s %6s | %s %6s" % (
            m, e["paper_fp32_fps"] or "-", "%.1f" % fp if fp else "-",
            "%.1f" % (fp / e["paper_fp32_fps"]) if fp and e["paper_fp32_fps"] else "-",
            f(a0["map50"] if a0 else None), f(a1["map50"] if a1 else None),
            "%+.4f" % (a1["map50"] - a0["map50"]) if a0 and a1 else "-",
            f(a1["far_ap50"] if a1 else None),
            "%+.4f" % (a1["far_ap50"] - a0["far_ap50"]) if a0 and a1 else "-"))
    json.dump(res, open(args.out, "w"), indent=1)
    print("-> %s" % args.out)


if __name__ == "__main__":
    main()
