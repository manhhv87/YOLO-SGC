# -*- coding: utf-8 -*-
"""export_e5.py - ONNX export for the TensorRT FP16 measurements on the Jetson Nano. Runs on a PC.

    python e5_nano/export_e5.py                 # 26 models -> e5_nano/onnx/<name>.onnx

Models (seed 0):
    12 unpruned    weights/main/<cfg>/best.pt       (Table 1)
    12 pruned      weights/prune/<cfg>/best.pt      (Table 7)
     2 n-scale     runs/nscale/<m>_s0/weights/best.pt  (YOLOv8n, YOLO11n)

Export protocol: batch 1, static 640x640, FP32, opset 12 (TensorRT 8.2 of JetPack 4.6 supports
opset <= 13), no NMS in the graph, then simplified with onnxslim. FP16 arithmetic is set by
trtexec --fp16 on the Nano, so the ONNX model stays in FP32.

Each checkpoint is copied to e5_nano/tmp/ and exported from there, so weights/ and runs/ are not changed.
onnx and onnxslim can also be placed in vendor/onnx_pkgs.
"""
import argparse, hashlib, json, os, shutil, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "vendor", "onnx_pkgs"))
sys.path.insert(0, ROOT)
os.environ["YOLO_AUTOINSTALL"] = "false"

import numpy as np

CFGS = ["yolov5s", "yolov5s_sgc_p345", "yolov5s_sgc_p45",
        "yolov8s", "yolov8s_sgc_p345", "yolov8s_sgc_p45",
        "yolov9s", "yolov9s_sgc_p345", "yolov9s_sgc_p45",
        "yolov11s", "yolov11s_sgc_p345", "yolov11s_sgc_p45"]


def model_list():
    out = []
    for c in CFGS:
        out.append((c, "weights/main/%s/best.pt" % c))
    for c in CFGS:
        out.append((c + "_pruned", "weights/prune/%s/best.pt" % c))
    for m in ("yolov8n", "yolov11n"):
        out.append((m, "runs/nscale/%s_s0/weights/best.pt" % m))
    return out


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="export only these names")
    ap.add_argument("--opset", type=int, default=12)
    args = ap.parse_args()

    import cv2, onnx, onnxslim, onnxruntime as ort, torch
    from ultralytics import YOLO
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import trt_common as tc

    out_dir = os.path.join(ROOT, "e5_nano", "onnx")
    tmp_dir = os.path.join(ROOT, "e5_nano", "tmp")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)
    test_img = sorted(os.listdir(os.path.join(ROOT, "datasets/test/images")))[0]
    bgr = cv2.imread(os.path.join(ROOT, "datasets/test/images", test_img))
    x, _, _ = tc.preprocess(bgr)

    man_path = os.path.join(out_dir, "manifest.json")
    manifest = json.load(open(man_path)) if os.path.exists(man_path) else {}
    for name, rel in model_list():
        if args.only and name not in args.only:
            continue
        src = os.path.join(ROOT, rel)
        t0 = time.time()
        tmp_pt = os.path.join(tmp_dir, name + "_e5.pt")  # the suffix keeps Ultralytics from mapping "*yolov5s.pt" to "yolov5su.pt"
        shutil.copy2(src, tmp_pt)
        f = YOLO(tmp_pt).export(format="onnx", imgsz=640, batch=1, opset=args.opset, simplify=False,
                                dynamic=False, half=False, nms=False, device="cpu", verbose=False)
        m = onnx.load(f)
        m = onnxslim.slim(m)
        onnx.checker.check_model(m)
        dst = os.path.join(out_dir, name + ".onnx")
        onnx.save(m, dst)
        os.remove(f)

        # compare ONNX (onnxruntime CPU) with PyTorch FP32 on one test image
        net = YOLO(tmp_pt).model.float().eval()
        net = net.fuse() if hasattr(net, "fuse") else net
        with torch.no_grad():
            y = net(torch.from_numpy(x))
        y = (y[0] if isinstance(y, (list, tuple)) else y).numpy()
        sess = ort.InferenceSession(dst, providers=["CPUExecutionProvider"])
        yo = sess.run(None, {sess.get_inputs()[0].name: x})[0]
        diff = float(np.abs(y - yo).max())
        os.remove(tmp_pt)

        manifest[name] = {
            "source": rel, "source_md5": md5(src), "onnx": "onnx/%s.onnx" % name,
            "onnx_mb": round(os.path.getsize(dst) / 1e6, 2), "opset": args.opset,
            "input": [d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim],
            "output": [d.dim_value for d in m.graph.output[0].type.tensor_type.shape.dim],
            "max_abs_diff_vs_torch": diff, "seconds": round(time.time() - t0, 1),
        }
        print("%-24s %6.1f MB  out=%s  |onnx-torch|max=%.2e  %.0fs" % (
            name, manifest[name]["onnx_mb"], manifest[name]["output"], diff, manifest[name]["seconds"]), flush=True)
        json.dump(manifest, open(man_path, "w"), indent=1)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print("-> %s (%d models)" % (man_path, len(manifest)))


if __name__ == "__main__":
    main()
