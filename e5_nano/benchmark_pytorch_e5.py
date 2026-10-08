"""
benchmark_pytorch_e5.py - FP32 throughput on the Jetson Nano with PyTorch (CUDA, TorchScript).

Same protocol as the Nano FP32 columns of Tables 1 and 7: 640x640 input, 20 warm-up passes, 100
timed passes with CUDA synchronization, 5% trimmed at each end, FPS = 1000 / mean latency in ms.

Models: YOLOv8n and YOLO11n (seed 0, exported on the PC to torchscript/ in this folder), plus two
models timed in the earlier session as controls (YOLOv8s: 220.55 ms; pruned YOLOv8s-SGC-P45:
36.22 ms), read from ~/models. YOLOv8s is timed again at the end to check that the conditions did
not change during the session.

Run on the Nano: /usr/bin/python3 benchmark_pytorch_e5.py
"""

import os
import time
from pathlib import Path

import torch

IMGSZ = 640
N_WARMUP = 20
N_RUNS = 100
DEVICE = "cuda:0"
HERE = Path(__file__).resolve().parent
HOME_MODELS = Path.home() / "models"
RESULT_FILE = HERE / "results_nano" / "fps_jetson_e5.csv"
REFERENCE = {"yolov8s": (220.55, 4.5), "yolov8s_sgc_p45_prune": (36.22, 27.6)}   # earlier session

MODELS = [HOME_MODELS / "yolov8s.torchscript",
          HOME_MODELS / "yolov8s_sgc_p45_prune.torchscript",
          HERE / "torchscript" / "yolov8n.torchscript",
          HERE / "torchscript" / "yolov11n.torchscript",
          HOME_MODELS / "yolov8s.torchscript"]

print("PyTorch {}".format(torch.__version__))
print("CUDA: {}".format(torch.cuda.get_device_name(0)))
print("Warmup: {}, Runs: {}, ImgSize: {}".format(N_WARMUP, N_RUNS, IMGSZ))
print("=" * 60)

results = []
for ts_path in MODELS:
    name = ts_path.stem
    print("Benchmarking: {}...".format(name), end=" ", flush=True)
    if not ts_path.exists():
        print("missing file {}".format(ts_path))
        continue
    try:
        model = torch.jit.load(str(ts_path), map_location=DEVICE)
        model.eval()
        model = model.to(DEVICE)
        dummy = torch.randn(1, 3, IMGSZ, IMGSZ).to(DEVICE)
        with torch.no_grad():
            for _ in range(N_WARMUP):
                model(dummy)
        torch.cuda.synchronize()
        latencies = []
        with torch.no_grad():
            for _ in range(N_RUNS):
                torch.cuda.synchronize()
                t0 = time.time()
                model(dummy)
                torch.cuda.synchronize()
                t1 = time.time()
                latencies.append((t1 - t0) * 1000)
        latencies = sorted(latencies)
        trim = int(N_RUNS * 0.05)
        if trim > 0:
            latencies = latencies[trim:-trim]
        avg_ms = sum(latencies) / len(latencies)
        fps = 1000.0 / avg_ms
        ref = REFERENCE.get(name)
        note = "" if ref is None else "  (earlier: {:.2f} ms / {:.1f} FPS)".format(*ref)
        print("{:.1f} ms | FPS: {:.1f}{}".format(avg_ms, fps, note))
        results.append({"name": name, "avg_ms": avg_ms, "fps": fps})
    except Exception as e:
        print("FAIL: {}".format(e))
        results.append({"name": name, "avg_ms": 0, "fps": 0})
    try:
        del model
    except Exception:
        pass
    torch.cuda.empty_cache()
    time.sleep(3)

if not RESULT_FILE.parent.exists():
    os.makedirs(str(RESULT_FILE.parent))
with open(str(RESULT_FILE), "w") as f:
    f.write("Model,Latency_ms,FPS\n")
    for r in results:
        f.write("{},{:.2f},{:.1f}\n".format(r["name"], r["avg_ms"], r["fps"]))
print("\nSaved: {}".format(RESULT_FILE))
