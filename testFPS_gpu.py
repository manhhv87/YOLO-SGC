# -*- coding: utf-8 -*-
"""testFPS_gpu.py - GPU throughput of every checkpoint in weights/.

RTX 5060 Ti, batch 1, FP32, 640x640, fused model; 100 warm-up iterations, then FPS from the
mean latency of 1000 iterations. Every model is measured in two independent passes (reloaded
each time).

    python testFPS_gpu.py                          # 2 passes, 100 + 1000 iterations
    python testFPS_gpu.py --passes 3 --iters 500

Output: paper_results/fps_gpu_remeasured.csv, written after every model.
"""
import argparse, csv, glob, os, statistics, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # use the local Ultralytics fork
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default="weights")
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--warmup", type=int, default=100)
ap.add_argument("--iters", type=int, default=1000)
ap.add_argument("--passes", type=int, default=2)
ap.add_argument("--out", default="paper_results/fps_gpu_remeasured.csv")
args = ap.parse_args()

assert torch.cuda.is_available(), "a CUDA GPU is required"
dev = torch.device("cuda:0")
gpu_name = torch.cuda.get_device_name(0)
print(f"GPU: {gpu_name} | torch {torch.__version__} | cudnn.benchmark={torch.backends.cudnn.benchmark}")

pts = sorted(glob.glob(os.path.join(args.weights, "*", "*", "best.pt")))
if not pts:
    raise SystemExit(f"no best.pt found in {args.weights}/")
print(f"{len(pts)} checkpoint x {args.passes} pass, {args.warmup}+{args.iters} iterations per pass\n")


def load_fused(pt):
    ck = torch.load(pt, map_location="cpu", weights_only=False)
    m = (ck.get("model") or ck.get("ema")).float().eval()  # some checkpoints store only the EMA model
    for p in m.parameters():
        p.requires_grad_(False)
    try:
        m.fuse(verbose=False)
    except TypeError:
        m.fuse()
    return m.to(dev)


@torch.inference_mode()
def bench(m):
    x = torch.randn(1, 3, args.imgsz, args.imgsz, device=dev)
    for _ in range(args.warmup):
        m(x)
    torch.cuda.synchronize()
    lat = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        m(x)
        torch.cuda.synchronize()
        lat.append(time.perf_counter() - t0)
    mean_ms = statistics.mean(lat) * 1e3
    med_ms = statistics.median(lat) * 1e3
    return 1e3 / mean_ms, 1e3 / med_ms, mean_ms


os.makedirs(os.path.dirname(args.out), exist_ok=True)
rows = {}  # key -> dict
t_start = time.time()
for pss in range(1, args.passes + 1):
    print(f"===== PASS {pss}/{args.passes} =====")
    for pt in pts:
        p = pt.replace("\\", "/")
        key = p.split("/")[-3] + "/" + p.split("/")[-2]
        try:
            m = load_fused(pt)
            npar = sum(q.numel() for q in m.parameters()) / 1e6
            fps_mean, fps_med, ms = bench(m)
            del m
            torch.cuda.empty_cache()
            r = rows.setdefault(key, {"model": key, "Params_fused(M)": round(npar, 3)})
            r[f"FPS_pass{pss}"] = round(fps_mean, 1)
            r[f"FPSmed_pass{pss}"] = round(fps_med, 1)
            print(f"  {key:34s} {fps_mean:7.1f} FPS (median {fps_med:7.1f}, {ms:6.2f} ms, {npar:6.3f}M)")
        except Exception as e:
            print(f"  ERR {key}: {e}")
        # written after every model so that no result is lost
        cols = ["model", "Params_fused(M)"] + [c for i in range(1, args.passes + 1) for c in (f"FPS_pass{i}", f"FPSmed_pass{i}")]
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for k in sorted(rows):
                w.writerow({c: rows[k].get(c, "") for c in cols})

print(f"\ndone in {(time.time()-t_start)/60:.1f} min -> {args.out}")
print(f"GPU: {gpu_name} | torch {torch.__version__} | FP32 | imgsz {args.imgsz} | warmup {args.warmup} | iters {args.iters}")
