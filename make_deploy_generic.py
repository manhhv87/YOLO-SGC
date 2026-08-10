# -*- coding: utf-8 -*-
"""make_deploy_generic.py — build the deployment variant (ScaleMap branch removed) for any backbone.

The four backbones have different layer maps (v5s/v8s: ScaleMap at 12-14, two SGCBlocks;
v11s: 13-15; v9s: 20-21 with a single SGCBlock), so a hard-coded index table will not do.

Procedure:
  1. drop every ScaleMapHead / ScaleMapDown layer;
  2. renumber the remaining layers and rewrite every 'from' reference;
  3. swap SGCBlock -> SGCBlock_Deploy and drop the scale map from its input list;
  4. port the weights: rename keys to the new indices and cut the trailing columns of
     weight_conv, which is mathematically equivalent to setting S = 0.

    python make_deploy_generic.py --src weights/main/yolov5s_sgc_p345/best.pt \
        --out weights/deploy/yolov5s_sgc_p345_deploy --name yolov5s_sgc_p345_deploy
"""
import argparse, os, sys
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="trained checkpoint")
ap.add_argument("--out", required=True, help="output directory for the generated .yaml and .pt")
ap.add_argument("--name", required=True, help="yaml name to write under ultralytics/cfg/models/v8/")
args = ap.parse_args()

DROP = {"ScaleMapHead", "ScaleMapDown"}


def transform(d):
    """Return (new yaml, old->new index map, set of dropped indices)."""
    L = d["backbone"] + d["head"]
    nb = len(d["backbone"])
    drop = {i for i, l in enumerate(L) if l[2] in DROP}
    keep = [i for i in range(len(L)) if i not in drop]
    imap = {old: new for new, old in enumerate(keep)}

    def remap(f, is_sgc):
        if isinstance(f, int):
            if f == -1:
                return -1
            assert f not in drop, f"a kept layer references a dropped layer: {f}"
            return imap[f]
        out = [x for x in f if not (is_sgc and x in drop)]
        for x in out:
            assert x not in drop, f"reference to a dropped layer: {x}"
        return [(-1 if x == -1 else imap[x]) for x in out]

    new = []
    for i in keep:
        frm, rep, mod, a = L[i][0], L[i][1], L[i][2], L[i][3] if len(L[i]) > 3 else []
        is_sgc = str(mod).startswith("SGCBlock")
        # -1 is valid only if the immediately preceding layer is kept as well
        if frm == -1 or (isinstance(frm, list) and -1 in frm):
            assert i - 1 not in drop, f"layer {i} uses -1 but the preceding layer was dropped"
        new.append([remap(frm, is_sgc), rep, "SGCBlock_Deploy" if is_sgc else mod, a])

    nb_new = sum(1 for i in keep if i < nb)
    out = {k: v for k, v in d.items() if k not in ("backbone", "head")}
    out["nc"] = d.get("nc", 4)
    out["backbone"], out["head"] = new[:nb_new], new[nb_new:]
    return out, imap, drop


def main():
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_num_params, get_flops

    ck = torch.load(args.src, map_location="cpu", weights_only=False)
    src = ck["model"]
    d_new, imap, drop = transform(src.yaml)

    ypath = f"ultralytics/cfg/models/v8/{args.name}.yaml"
    yaml.safe_dump(d_new, open(ypath, "w"), sort_keys=False, default_flow_style=None)

    net = YOLO(ypath).model.float()
    # The prior channel count DIFFERS between backbones (v8s/v9s: 1, v11s: 8), so we cannot
    # cut a fixed column; cut exactly what the deployment model expects. The map is
    # concatenated LAST, so the feature columns come first.
    want = {n: m.in_channels for n, m in net.named_modules() if n.endswith("weight_conv")}

    sd, new_sd = src.float().state_dict(), {}
    for k, v in sd.items():
        p = k.split(".")
        if p[0] == "model" and p[1].isdigit():
            i = int(p[1])
            if i in drop:
                continue
            k = ".".join(["model", str(imap[i])] + p[2:])
            if k.endswith("weight_conv.weight"):
                nin = want[k[:-len(".weight")]]
                assert v.shape[1] > nin, f"{k}: {v.shape[1]} -> {nin}"
                v = v[:, :nin].contiguous()      # drop exactly the prior channels
        new_sd[k] = v

    miss, unexp = net.load_state_dict(new_sd, strict=False)
    miss = [m for m in miss if "num_batches_tracked" not in m]
    unexp = [u for u in unexp if "num_batches_tracked" not in u]
    assert not miss and not unexp, f"state_dict mismatch: missing {miss[:3]} unexpected {unexp[:3]}"

    net.nc, net.names = src.nc, src.names
    ck["model"] = net
    ck.pop("ema", None); ck.pop("optimizer", None)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(ck, args.out + ".pt")

    p0, f0 = get_num_params(src), get_flops(src, 640)
    p1, f1 = get_num_params(net), get_flops(net, 640)
    print(f"{os.path.basename(args.src.rstrip('/'))}: {p0/1e6:.3f} M / {f0:.2f} G  ->  "
          f"{p1/1e6:.3f} M / {f1:.2f} G   ({len(drop)} layers dropped, -{100*(f0-f1)/f0:.1f}% GFLOPs)")
    print(f"-> {ypath}\n-> {args.out}.pt")


if __name__ == "__main__":
    main()
