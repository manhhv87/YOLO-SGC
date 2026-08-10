# -*- coding: utf-8 -*-
"""make_deploy_ckpt.py — port a trained checkpoint to the deployment architecture (no ScaleMapHead).

  YPrior  -> yolov8s_sgc_p345_yprior_deploy.yaml : weights transfer UNCHANGED
             (the block generates its own ramp; the ScaleMapHead output was already ignored)
  Full    -> yolov8s_sgc_p345_deploy.yaml        : cut the trailing column of weight_conv
             (mathematically equivalent to setting S = 0)

Layers 12/13/14 are dropped, so every index from 15 onwards shifts down by three.

    python make_deploy_ckpt.py --src runs/.../best.pt --kind yprior --out weights/deploy/yprior.pt
"""
import argparse, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_deploy_yaml import INDEX_MAP, DROPPED  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--kind", choices=["yprior", "full"], required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--fold_const", type=int, default=0, metavar="N",
                help="with --kind full: fold the mean of S (measured over N images) into the bias "
                     "of weight_conv instead of setting S=0. More accurate, because a constant "
                     "channel contributes a constant added to the logits.")
args = ap.parse_args()

YAML = {"yprior": "yolov8s_sgc_p345_yprior_deploy.yaml", "full": "yolov8s_sgc_p345_deploy.yaml"}


def remap(sd, cut_last_col):
    out, skipped = {}, 0
    for k, v in sd.items():
        parts = k.split(".")
        if parts[0] != "model" or not parts[1].isdigit():
            out[k] = v
            continue
        i = int(parts[1])
        if i in DROPPED:
            skipped += 1
            continue
        j = INDEX_MAP.get(i, i)
        nk = ".".join(["model", str(j)] + parts[2:])
        if cut_last_col and nk.endswith("weight_conv.weight"):
            v = v[:, :-1].contiguous()          # drop the prior channel column
        out[nk] = v
    return out, skipped


def measure_mean_S(src_pt, n):
    """Spatial mean of the prior channel at each SGCBlock, over n test images."""
    import glob
    from ultralytics import YOLO
    m = YOLO(src_pt)
    acc = {}
    for name, mod in m.model.model.named_modules():
        if mod.__class__.__name__.startswith("SGCBlock"):
            def pre(md, inp, name=name):
                acc.setdefault(name, []).append(float(inp[0][:, -1:].mean()))
            mod.weight_conv.register_forward_pre_hook(pre)
    for p in sorted(glob.glob("datasets/test/images/*"))[:n]:
        m.predict(p, imgsz=640, verbose=False)
    return {k: sum(v) / len(v) for k, v in acc.items()}


def main():
    from ultralytics import YOLO
    ck = torch.load(args.src, map_location="cpu", weights_only=False)
    src_model = ck["model"]
    sd = src_model.float().state_dict()

    if args.kind == "full" and args.fold_const:
        cs = measure_mean_S(args.src, args.fold_const)
        print("mean S per block:", {k: round(v, 4) for k, v in cs.items()})
        for name, c in cs.items():
            w = sd[f"model.{name}.weight_conv.weight"]        # (3, 3C+1, 1, 1)
            sd[f"model.{name}.weight_conv.bias"] = (
                sd[f"model.{name}.weight_conv.bias"] + w[:, -1, 0, 0] * c)

    new = YOLO(YAML[args.kind]).model.float()
    new_sd, skipped = remap(sd, cut_last_col=(args.kind == "full"))
    missing, unexpected = new.load_state_dict(new_sd, strict=False)
    missing = [m for m in missing if "num_batches_tracked" not in m]
    unexpected = [u for u in unexpected if "num_batches_tracked" not in u]
    print(f"dropped {skipped} ScaleMapHead tensors | missing {len(missing)} | unexpected {len(unexpected)}")
    if missing[:5]:
        print("  missing:", missing[:5])
    if unexpected[:5]:
        print("  unexpected:", unexpected[:5])
    assert not missing and not unexpected, "state_dict mismatch, aborting"

    new.nc = src_model.nc
    new.names = src_model.names
    new.args = getattr(src_model, "args", None)
    ck["model"] = new.half() if next(src_model.parameters()).dtype == torch.float16 else new
    ck.pop("ema", None)
    ck.pop("optimizer", None)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(ck, args.out)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
