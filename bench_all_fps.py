# -*- coding: utf-8 -*-
"""bench_all_fps.py — measure FPS for every model under a single protocol.

Protocol used in the paper: batch 1, FP32, 640x640, fused model, 100 warm-up iterations,
FPS = iterations / time over 1000 iterations, timed with CUDA events on an idle GPU.

Measurement design, established by diagnosis:
  * Measuring one model ten times in a row: 293.8 -> 293.4 FPS, 0.1% drift although the
    temperature rose 65->74C, so time drift is negligible and round-robin interleaving
    is unnecessary.
  * But holding several models on the GPU at once shifts the reading by up to 2.5%,
    depending on which models are co-resident, so each model is measured ALONE and freed
    before the next is loaded. The GPU is warmed first to leave the idle clock (180 ->
"""
import argparse, gc, json, os, statistics, subprocess, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=3, help="repetitions per model (median is reported)")
ap.add_argument("--warmup", type=int, default=100)
ap.add_argument("--iters", type=int, default=1000)
ap.add_argument("--prewarm", type=int, default=2000)
ap.add_argument("--out", default="paper_results/fps_all_rm5.json")
args = ap.parse_args()

ANCHOR = "weights/main/yolov8s/best.pt"
BB = ["yolov5s", "yolov8s", "yolov9s", "yolov11s"]
MODELS = [(f"T1 {b}{v}", f"weights/main/{b}{v}/best.pt")
          for b in BB for v in ["", "_sgc_p345", "_sgc_p45"]]
MODELS += [(f"T5 pruned {b}{v}", f"weights/prune/{b}{v}/best.pt")
           for b in BB for v in ["", "_sgc_p345", "_sgc_p45"]]
MODELS += [("T3 AMRF+FPN", "weights/ablation/AMRF_FPN/best.pt"),
           ("T3 Minimal", "weights/ablation/minimal/best.pt"),
           ("T3 w/o AMRF", "weights/ablation/NoAMRF/best.pt"),
           ("T3 w/o ScaleMap", "weights/ablation/NoScale/best.pt"),
           ("T3 w/o Adpt.W.", "weights/ablation/EqualWeight/best.pt"),
           ("T3 YPrior", "runs/ablation2/yolov8s_sgc_p345_yprior_s0/weights/best.pt"),
           ("DEPLOY yprior", "weights/deploy/yprior_s0_deploy.pt"),
           ("DEPLOY full", "weights/deploy/full_s0_deploy.pt")]


def gpu():
    return subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm",
                           "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()


def load(pt):
    from ultralytics import YOLO
    m = YOLO(pt).model.float().eval().cuda()
    return m.fuse() if hasattr(m, "fuse") else m


def timed(m, x):
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
    return args.iters / (s.elapsed_time(e) / 1000.0)


def measure(pt, x):
    """Load, measure, free completely. No other model stays resident on the GPU."""
    m = load(pt)
    v = [timed(m, x) for _ in range(args.reps)]
    del m
    gc.collect()
    torch.cuda.empty_cache()
    return v


def main():
    busy = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()
    if busy:
        print(f"STOP: another process is using the GPU ({busy}).")
        return
    avail = [(n, p) for n, p in MODELS if os.path.exists(p)]
    x = torch.rand(1, 3, 640, 640, device="cuda")
    print(f"Measuring {len(avail)} models one at a time, {args.reps} reps each. GPU: {gpu()}")

    warm = load(ANCHOR)
    with torch.no_grad():
        for _ in range(args.prewarm):
            warm(x)
    torch.cuda.synchronize()
    del warm
    gc.collect()
    torch.cuda.empty_cache()
    a0 = statistics.median(measure(ANCHOR, x))
    print(f"Session START anchor: {a0:.1f} FPS   GPU: {gpu()}")

    res = {}
    for n, p in avail:
        v = measure(p, x)
        res[n] = {"fps": statistics.median(v), "raw": v}
        print(f"  {n:28s} {res[n]['fps']:7.1f} FPS   (spread {100*(max(v)-min(v))/statistics.median(v):.1f}%)")

    a1 = statistics.median(measure(ANCHOR, x))
    drift = 100 * (a1 - a0) / a0
    print(f"\nSession END anchor: {a1:.1f} FPS   -> drift over the session {drift:+.2f}%")
    print("OK: measurement conditions stable." if abs(drift) <= 1 else "WARNING: conditions changed.")
    print(f"GPU after the run: {gpu()}")
    json.dump({"anchor_start": a0, "anchor_end": a1, "drift_pct": drift,
               "results": res, "args": vars(args)}, open(args.out, "w"), indent=1)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
