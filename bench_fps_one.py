# -*- coding: utf-8 -*-
"""bench_fps_one.py — measure FPS for exactly one model in a clean process, printing JSON.

Used with bench_fps_all.sh: one process per model, so no memory fragmentation accumulates
across models. Diagnosis: loading and freeing 32 models in one process cost 2.07% of
throughput, whereas repeating the same model drifted only 0.1%.

Protocol: batch 1, FP32, 640, fused, GPU warmed, 100 warm-up plus 1000 timed iterations, CUDA events.
"""
import argparse, json, os, statistics, subprocess, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--name", default="")
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--warmup", type=int, default=100)
ap.add_argument("--iters", type=int, default=1000)
ap.add_argument("--prewarm", type=int, default=1500)
args = ap.parse_args()


def gpu():
    return subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm",
                           "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()


def main():
    from ultralytics import YOLO
    m = YOLO(args.ckpt).model.float().eval().cuda()
    m = m.fuse() if hasattr(m, "fuse") else m
    x = torch.rand(1, 3, 640, 640, device="cuda")
    with torch.no_grad():
        for _ in range(args.prewarm):        # warm up to leave the idle clock
            m(x)
    torch.cuda.synchronize()

    v = []
    for _ in range(args.reps):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        with torch.no_grad():
            for _ in range(args.warmup):
                m(x)
            torch.cuda.synchronize()
            s.record()
            for _ in range(args.iters):
                m(x)
            e.record()
            torch.cuda.synchronize()
        v.append(args.iters / (s.elapsed_time(e) / 1000.0))

    print("JSON " + json.dumps({"name": args.name or os.path.basename(os.path.dirname(args.ckpt)),
                                "ckpt": args.ckpt, "fps": statistics.median(v),
                                "spread_pct": 100 * (max(v) - min(v)) / statistics.median(v),
                                "raw": v, "gpu": gpu()}))


if __name__ == "__main__":
    main()
