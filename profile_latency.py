# -*- coding: utf-8 -*-
"""profile_latency.py — why can fewer parameters and fewer GFLOPs still run slower?

Times each top-level layer (model.model[i]) with CUDA events, groups them by module type,
and compares against the FLOPs of that layer to give an arithmetic intensity in FLOPs per
millisecond. A layer with much time but few FLOPs is the one bound by memory bandwidth.

    python profile_latency.py --a weights/main/yolov5s/best.pt --b weights/main/yolov5s_sgc_p345/best.pt
"""
import argparse, collections, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--a", required=True, help="reference checkpoint")
ap.add_argument("--b", required=True, help="checkpoint to compare")
ap.add_argument("--iters", type=int, default=200)
ap.add_argument("--warmup", type=int, default=50)
args = ap.parse_args()


def load(pt):
    from ultralytics import YOLO
    m = YOLO(pt).model.float().eval().cuda()
    return m.fuse() if hasattr(m, "fuse") else m


def profile(pt):
    net = load(pt)
    x = torch.rand(1, 3, 640, 640, device="cuda")
    layers = list(net.model)
    ev = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in layers]
    tot = collections.defaultdict(float)
    cnt = collections.Counter()

    hooks = []
    for i, m in enumerate(layers):
        hooks.append(m.register_forward_pre_hook(lambda mod, inp, i=i: ev[i][0].record()))
        hooks.append(m.register_forward_hook(lambda mod, inp, out, i=i: ev[i][1].record()))

    with torch.no_grad():
        for _ in range(args.warmup):
            net(x)
        torch.cuda.synchronize()
        for _ in range(args.iters):
            net(x)
            torch.cuda.synchronize()
            for i, m in enumerate(layers):
                tot[i] += ev[i][0].elapsed_time(ev[i][1])
    for h in hooks:
        h.remove()

    # count the operator types inside
    dw = sum(1 for _, m in net.named_modules()
             if isinstance(m, torch.nn.Conv2d) and m.groups > 1)
    conv = sum(1 for _, m in net.named_modules() if isinstance(m, torch.nn.Conv2d))
    by_class = collections.defaultdict(float)
    for i, m in enumerate(layers):
        by_class[m.__class__.__name__] += tot[i] / args.iters
    return {"total_ms": sum(tot.values()) / args.iters, "by_class": dict(by_class),
            "n_layers": len(layers), "n_conv": conv, "n_depthwise": dw,
            "per_layer": {i: tot[i] / args.iters for i in range(len(layers))},
            "classes": [m.__class__.__name__ for m in layers]}


def main():
    from ultralytics.utils.torch_utils import get_num_params, get_flops
    out = {}
    for tag, pt in (("A", args.a), ("B", args.b)):
        r = profile(pt)
        n = load(pt)
        r["params"] = get_num_params(n) / 1e6
        r["gflops"] = get_flops(n, 640)
        out[tag] = r
        print(f"\n=== {tag}: {os.path.dirname(pt)} ===")
        print(f"  {r['params']:.2f} M | {r['gflops']:.1f} GFLOPs | total {r['total_ms']:.3f} ms "
              f"({1000/r['total_ms']:.0f} layers/s equivalent)")
        print(f"  {r['n_layers']} top-level layers | {r['n_conv']} conv, of which "
              f"{r['n_depthwise']} depthwise ({100*r['n_depthwise']/max(r['n_conv'],1):.0f}%)")
        print("  time by layer type (ms, shown only above 2% of total):")
        for k, v in sorted(r["by_class"].items(), key=lambda x: -x[1]):
            if v / r["total_ms"] > 0.02:
                print(f"    {k:22s} {v:7.3f} ms  {100*v/r['total_ms']:5.1f}%")
    a, b = out["A"], out["B"]
    print(f"\n=== SO SÁNH ===")
    print(f"  params   {a['params']:.2f} -> {b['params']:.2f} M  ({100*(b['params']-a['params'])/a['params']:+.1f}%)")
    print(f"  GFLOPs   {a['gflops']:.1f} -> {b['gflops']:.1f}      ({100*(b['gflops']-a['gflops'])/a['gflops']:+.1f}%)")
    print(f"  time     {a['total_ms']:7.3f} -> {b['total_ms']:.3f} ms ({100*(b['total_ms']-a['total_ms'])/a['total_ms']:+.1f}%)")
    print(f"  GFLOPs/ms {a['gflops']/a['total_ms']:.1f} -> {b['gflops']/b['total_ms']:.1f}  "
          f"(arithmetic intensity; lower means more memory-bound)")
    print(f"  layers   {a['n_layers']} -> {b['n_layers']}   depthwise {a['n_depthwise']} -> {b['n_depthwise']}")


if __name__ == "__main__":
    main()
