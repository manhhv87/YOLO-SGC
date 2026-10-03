# -*- coding: utf-8 -*-
"""train_seed_prune.py - seeds 1-4 of the pruned models of Table 7.

Each run builds the seed-0 architecture before pruning from e4_<cfg>_prune.yaml (written from the
model.yaml stored in the seed-0 pruned checkpoint), prunes it with Torch-Pruning (L2 magnitude) and
trains it with the recipe of train_seed.py. Every conv layer that kept its width in the seed-0 pruned
checkpoint (--ref) is kept here as well, so all five runs share the pruned architecture of Table 7
(e4_eval.py records this as same_arch_as_seed0).

    python train_seed_prune.py --yaml e4_yolov8s_sgc_p345_prune.yaml --weights yolov8s.pt \
        --ref weights/prune/yolov8s_sgc_p345/best.pt \
        --data ultralytics/cfg/datasets/corn.yaml --seed 1 --name yolov8s_sgc_p345_s1 --amp
"""
import argparse

import torch
import torch.nn as nn
import torch_pruning as tp
from ultralytics import YOLO

# ---- RECIPE: copied from train_seed.py ----
RECIPE = dict(
    optimizer="auto", cos_lr=True, deterministic=True,
    lr0=0.005, lrf=0.01, momentum=0.937, weight_decay=0.0005,
    warmup_epochs=5.0, warmup_momentum=0.8, warmup_bias_lr=0.1,
    box=7.5, cls=0.5, dfl=1.5,
    patience=30, close_mosaic=20, max_det=50,
    # augmentation
    hsv_h=0.015, hsv_s=0.5, hsv_v=0.3,
    degrees=0.0, translate=0.03, scale=0.2, shear=0.0, perspective=0.0,
    flipud=0.0, fliplr=0.5, mosaic=0.1, mixup=0.0, copy_paste=0.0, erasing=0.0,
)

ap = argparse.ArgumentParser()
ap.add_argument("--yaml", required=True, help="e4_<cfg>_prune.yaml (seed-0 architecture before pruning)")
ap.add_argument("--weights", required=True, help="COCO *.pt to load partially, as in the seed-0 run")
ap.add_argument("--ref", required=True, help="pruned seed-0 checkpoint (weights/prune/<cfg>/best.pt)")
ap.add_argument("--data", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--dir", default="runs/prune5")
ap.add_argument("--ratio", type=float, default=0.5)
ap.add_argument("--epochs", type=int, default=200)
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--device", default="0")
ap.add_argument("--amp", action="store_true")
args = ap.parse_args()

# skip channel groups whose importance Torch-Pruning cannot compute (left unpruned, as in seed 0)
_orig_call = tp.importance.MagnitudeImportance.__call__


def _safe_call(self, group, *a, **k):
    try:
        return _orig_call(self, group, *a, **k)
    except IndexError:
        return None


tp.importance.MagnitudeImportance.__call__ = _safe_call

# a layer that kept its output width in the seed-0 pruned model is kept here as well
_ref = torch.load(args.ref, map_location="cpu", weights_only=False)
_ref_sd = (_ref.get("model") or _ref.get("ema")).float().state_dict()
_orig_init = tp.pruner.MagnitudePruner.__init__


def _init_keep_like_seed0(self, model, example_inputs, *a, ignored_layers=None, **k):
    ign = list(ignored_layers or [])
    for n, x in model.named_modules():
        w = _ref_sd.get(f"{n}.weight")
        if isinstance(x, nn.Conv2d) and w is not None and w.shape[0] == x.out_channels:
            ign.append(x)
    _orig_init(self, model, example_inputs, *a, ignored_layers=ign, **k)


tp.pruner.MagnitudePruner.__init__ = _init_keep_like_seed0

model = YOLO(args.yaml)
model.load(args.weights)                 # partial COCO load (matching keys and shapes)
model.train(
    data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
    device=args.device, seed=args.seed, name=args.name, project=args.dir,
    amp=args.amp, workers=8, verbose=True,
    prune=True, prune_ratio=args.ratio, prune_iterative_steps=1,
    **RECIPE,
)
print("DONE:", args.name)
