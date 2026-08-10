# -*- coding: utf-8 -*-
"""scalemap_intervention.py — causal intervention experiment on the ScaleMap.

Overwrites the ScaleMapHead output at inference (a single intervention point, since
ScaleMapDown only pools this map for P3/P4), then measures per-distance-band AP on the
test set. Weights are NOT retrained.

Interventions:
  none      — unchanged (control, must match the pre-pruning table);
  const     — replaced by its own spatial mean (removes all spatial structure,
              keeps the magnitude);
  vflip     — vertical flip (keeps histogram and structure, only swaps far and near);
  hflip     — horizontal flip (CONTROL: keeps the vertical order, breaks alignment with
              image content) — if vflip hurts more than hflip, the vertical axis is what matters;
  rowmean   — each row replaced by its own mean (keeps the vertical profile, removes
              the soil/vegetation component entirely);
  ramp      — replaced by a linear vertical gradient, rescaled to the [min,max] of the
              original map for each image (an ideal, parameter-free vertical prior).

    python scalemap_intervention.py --out paper_results/scalemap_intervention.json
"""
import argparse, json, os
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="weights/main/yolov8s_sgc_p345/best.pt")
ap.add_argument("--data", default="ultralytics/cfg/datasets/corn.yaml")
ap.add_argument("--split", default="test")
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--device", default="0")
ap.add_argument("--out", default="paper_results/scalemap_intervention.json")
ap.add_argument("--modes", default="none,const,vflip,hflip,rowmean,ramp")
ap.add_argument("--target", default="scalemap", choices=["scalemap", "fusion"],
                help="intervene on the scale map, or on the fusion weight logits themselves")
args = ap.parse_args()

BANDS = ["near", "mid_near", "mid_far", "far"]


def intervene(mode):
    """Return a function that transforms the scale-map tensor (B,1,H,W)."""
    if mode == "none":
        return None
    if mode == "const":
        return lambda s: s.mean(dim=(2, 3), keepdim=True).expand_as(s).contiguous()
    if mode == "vflip":
        return lambda s: torch.flip(s, dims=[2])
    if mode == "hflip":
        return lambda s: torch.flip(s, dims=[3])
    if mode == "rowmean":
        return lambda s: s.mean(dim=3, keepdim=True).expand_as(s).contiguous()
    if mode == "zero":                       # exactly the bypass of the w/o ScaleMap ablation row
        return lambda s: torch.zeros_like(s)
    if mode == "noise":                      # keeps mean/std, removes all structure
        return lambda s: (torch.randn_like(s) * s.std(dim=(2, 3), keepdim=True)
                          + s.mean(dim=(2, 3), keepdim=True))
    if mode == "shuffle":                    # shuffle the row order
        return lambda s: s[:, :, torch.randperm(s.shape[2], device=s.device), :]
    if mode == "ramp":
        def f(s):
            b, c, h, w = s.shape
            lo = s.amin(dim=(2, 3), keepdim=True)
            hi = s.amax(dim=(2, 3), keepdim=True)
            r = torch.linspace(0, 1, h, device=s.device, dtype=s.dtype).view(1, 1, h, 1)
            return lo + (hi - lo) * r.expand(b, c, h, w)
        return f
    raise ValueError(mode)


def intervene_fusion(mode):
    """Intervene directly on the weight_conv logits z (before the softmax)."""
    if mode == "none":
        return None
    if mode == "uniform":                    # z := 0 -> uniform fusion weights of 1/3
        return lambda z: torch.zeros_like(z)
    if mode == "zmean":                      # removes SPATIAL variation, keeps the channel bias
        return lambda z: z.mean(dim=(2, 3), keepdim=True).expand_as(z).contiguous()
    if mode == "zshuffle":                   # shuffle the weights spatially
        def f(z):
            b, c, h, w = z.shape
            idx = torch.randperm(h * w, device=z.device)
            return z.flatten(2)[:, :, idx].view(b, c, h, w).contiguous()
        return f
    raise ValueError(mode)


def run(mode):
    from ultralytics import YOLO
    model = YOLO(args.ckpt)
    if args.target == "fusion":
        fn = intervene_fusion(mode)
        if fn is not None:
            n = 0
            for _, m in model.model.model.named_modules():
                if m.__class__.__name__.startswith("SGCBlock"):
                    m.weight_conv.register_forward_hook(lambda mod, i, o, fn=fn: fn(o))
                    n += 1
            if not n:
                raise RuntimeError("SGCBlock not found")
    else:
        fn = intervene(mode)
        if fn is not None:
            for _, m in model.model.model.named_modules():
                if m.__class__.__name__ == "ScaleMapHead":
                    m.register_forward_hook(lambda mod, inp, out, fn=fn: fn(out))
                    break
            else:
                raise RuntimeError("ScaleMapHead not found")
    r = model.val(data=args.data, imgsz=args.imgsz, split=args.split, device=args.device,
                  verbose=False, plots=False)
    per = {str(r.names[int(ci)]): float(r.box.ap50[i]) for i, ci in enumerate(r.box.ap_class_index)}
    return {"mAP50": float(r.box.map50), "mAP50_95": float(r.box.map),
            "P": float(r.box.mp), "R": float(r.box.mr), "per_class_ap50": per}


def main():
    res = {}
    for mode in args.modes.split(","):
        res[mode] = run(mode)
        r = res[mode]
        print(f"{mode:8s} mAP50={r['mAP50']:.4f}  " +
              "  ".join(f"{b}={r['per_class_ap50'].get(b, float('nan')):.3f}" for b in BANDS))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)

    base = res.get("none")
    if base:
        print("\nChange against the control (percentage points):")
        hdr = f"{'mode':8s} {'mAP50':>8s} " + " ".join(f"{b:>10s}" for b in BANDS)
        print(hdr)
        for mode, r in res.items():
            if mode == "none":
                continue
            d = [100 * (r["per_class_ap50"].get(b, np.nan) - base["per_class_ap50"].get(b, np.nan))
                 for b in BANDS]
            print(f"{mode:8s} {100*(r['mAP50']-base['mAP50']):+8.2f} " +
                  " ".join(f"{v:+10.2f}" for v in d))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
