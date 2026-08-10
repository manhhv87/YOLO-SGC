# -*- coding: utf-8 -*-
"""make_deploy_pruned.py — remove the ScaleMap branch from an already pruned model.

Rebuilding from the yaml is not possible: pruning removes channels unevenly, so the
original yaml no longer describes the channel counts. The loaded graph is edited directly:
  1. drop the ScaleMapHead / ScaleMapDown layers from the nn.Sequential;
  2. renumber .i and rewrite the .f references of every remaining layer;
  3. for each SGCBlock, switch the class to SGCBlock_Deploy (same submodules) and cut the
     prior channel columns from weight_conv, equivalent to setting S = 0;
  4. recompute the .save list of layers whose outputs must be kept.

    python make_deploy_pruned.py --src weights/prune/yolov8s_sgc_p345/best.pt \
        --out weights/deploy/pruned_yolov8s_sgc_p345_deploy.pt
"""
import argparse, os, sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

DROP = {"ScaleMapHead", "ScaleMapDown"}


def main():
    from ultralytics import YOLO
    from ultralytics.nn.modules.block import SGCBlock_Deploy
    from ultralytics.utils.torch_utils import get_num_params, get_flops

    ck = torch.load(args.src, map_location="cpu", weights_only=False)
    net = ck.get("model") or ck.get("ema")     # some pruned checkpoints only carry ema
    assert net is not None, "checkpoint contains neither model nor ema"
    net = net.float()
    layers = list(net.model)
    drop = {i for i, m in enumerate(layers) if m.__class__.__name__ in DROP}
    assert drop, "ScaleMap branch not found"
    keep = [i for i in range(len(layers)) if i not in drop]
    imap = {o: n for n, o in enumerate(keep)}

    # the scale-map channel count equals the ScaleMapHead output channels
    head = next(m for m in layers if m.__class__.__name__ == "ScaleMapHead")
    k = head.conv_out.out_channels

    new_layers = []
    for old_i in keep:
        m = layers[old_i]
        is_sgc = m.__class__.__name__.startswith("SGCBlock")
        f = m.f
        if isinstance(f, list):
            f = [x for x in f if x not in drop]
            f = [(-1 if x == -1 else imap[x]) for x in f]
        elif f != -1:
            f = imap[f]
        m.f, m.i = f, imap[old_i]
        if is_sgc:
            wc = m.weight_conv
            new_wc = nn.Conv2d(wc.in_channels - k, wc.out_channels, 1)
            with torch.no_grad():
                new_wc.weight.copy_(wc.weight[:, :wc.in_channels - k])
                new_wc.bias.copy_(wc.bias)
            m.weight_conv = new_wc
            m.__class__ = SGCBlock_Deploy      # same submodules, forward takes three inputs
        new_layers.append(m)

    net.model = nn.Sequential(*new_layers)
    save = set()
    for m in new_layers:
        for x in ([m.f] if isinstance(m.f, int) else m.f):
            if x != -1:
                save.add(x)
    net.save = sorted(save)

    with torch.no_grad():
        y = net(torch.zeros(1, 3, 640, 640))
    assert y is not None
    p, g = get_num_params(net), get_flops(net, 640)
    ck["model"] = net
    ck.pop("ema", None); ck.pop("optimizer", None)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(ck, args.out)
    print(f"dropped {len(drop)} layers, cut {k} prior channels  ->  {p/1e6:.3f} M / {g:.2f} GFLOPs")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
