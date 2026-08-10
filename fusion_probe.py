# -*- coding: utf-8 -*-
"""fusion_probe.py — measure how much the scale prior contributes to the SGCBlock fusion weights.

Supports both architectures:
  * original SGCBlock     : S is one channel of the single 1x1 conv, so z_S = W[:,-1]*S is
    separated analytically; * SGCBlock_SGate/_SFiLM: the prior has its own branch, hooked at s_logit.

    python fusion_probe.py --ckpt weights/main/yolov8s_sgc_p345/best.pt
"""
import argparse, glob, json, os
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="weights/main/yolov8s_sgc_p345/best.pt")
ap.add_argument("--data_root", default="datasets")
ap.add_argument("--split", default="test")
ap.add_argument("--n", type=int, default=50)
ap.add_argument("--out", default="")
args = ap.parse_args()

rec = {}


def acc(name):
    return rec.setdefault(name, {k: [] for k in ("std_z", "std_zS", "std_w", "dw", "w_mean",
                                                 "corr_row", "S_range")})


def stats(name, z, z_S, S):
    w = torch.softmax(z, 1)
    w_noS = torch.softmax(z - z_S, 1)
    d = acc(name)
    d["std_z"].append(z.std(dim=(2, 3)).mean().item())
    d["std_zS"].append(z_S.std(dim=(2, 3)).mean().item())
    d["std_w"].append(w.std(dim=(2, 3)).mean(0).cpu().numpy())
    d["dw"].append((w - w_noS).abs().mean().item())
    d["w_mean"].append(w.mean(dim=(0, 2, 3)).cpu().numpy())
    d["S_range"].append((S.amax() - S.amin()).item())
    H = w.shape[2]
    row = torch.linspace(0, 1, H, device=w.device).view(1, 1, H, 1).expand_as(w)
    wc, rc = w - w.mean(dim=(2, 3), keepdim=True), row - row.mean(dim=(2, 3), keepdim=True)
    corr = (wc * rc).mean(dim=(2, 3)) / (wc.std(dim=(2, 3)) * rc.std(dim=(2, 3)) + 1e-9)
    d["corr_row"].append(corr.mean(0).cpu().numpy())


def main():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ultralytics import YOLO
    model = YOLO(args.ckpt)

    n_hook = 0
    for name, mod in model.model.model.named_modules():
        cls = mod.__class__.__name__
        if not cls.startswith("SGCBlock"):
            continue
        if hasattr(mod, "s_logit"):                     # architecture with a separate prior branch
            store = {}

            def h_feat(m, i, o, name=name, store=store):
                store["z_feat"] = o.detach()

            def h_s(m, i, o, name=name, store=store):
                z_S = o.detach()
                if "z_feat" in store:
                    stats(f"{name}[{cls}]", store["z_feat"] + z_S, z_S, i[0].detach())

            mod.weight_conv.register_forward_hook(h_feat)
            mod.s_logit.register_forward_hook(h_s)
        else:                                           # original SGCBlock: analytic separation
            def h(m, i, o, name=name, cls=cls):
                S = i[0][:, -1:].detach()
                W = m.weight.detach()
                z_S = (W[:, -1, 0, 0].view(1, 3, 1, 1) * S)
                stats(f"{name}[{cls}]", o.detach(), z_S, S)

            mod.weight_conv.register_forward_hook(h)
        n_hook += 1
    if not n_hook:
        raise RuntimeError("no SGCBlock found")

    for imgp in sorted(glob.glob(f"{args.data_root}/{args.split}/images/*"))[:args.n]:
        model.predict(imgp, imgsz=640, verbose=False)

    res = {}
    print(f"{os.path.dirname(args.ckpt)}   (n = {args.n} images)\n")
    for name, d in rec.items():
        sz, szs = float(np.mean(d["std_z"])), float(np.mean(d["std_zS"]))
        res[name] = {"std_z": sz, "std_zS": szs, "share_pct": 100 * szs / sz,
                     "dw": float(np.mean(d["dw"])),
                     "w_mean": np.mean(np.stack(d["w_mean"]), 0).tolist(),
                     "std_w": np.mean(np.stack(d["std_w"]), 0).tolist(),
                     "corr_row": np.mean(np.stack(d["corr_row"]), 0).tolist(),
                     "S_range": float(np.mean(d["S_range"]))}
        r = res[name]
        print(f"{name}")
        print(f"  spatial std of logits z = {sz:.4f} | prior share z_S = {szs:.4f} "
              f"-> {r['share_pct']:.1f}%")
        print(f"  |dw| when the prior part is removed = {r['dw']:.5f}")
        print(f"  mean w = {np.round(r['w_mean'], 3)} | spatial std = "
              f"{np.round(r['std_w'], 4)} | correlation with image row = {np.round(r['corr_row'], 3)}\n")
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=1)
        print("->", args.out)


if __name__ == "__main__":
    main()
