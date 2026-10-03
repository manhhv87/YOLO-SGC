# -*- coding: utf-8 -*-
"""train_seed.py - train one configuration with a given seed, using the recipe of the original runs.

The model is built from its yaml, and the COCO weights are loaded where layer names and shapes match.

    python train_seed.py --model paper_yolov8s_sgc_p345.yaml --weights yolov8s.pt \
        --data ultralytics/cfg/datasets/corn.yaml --seed 1 --name yolov8s_sgc_p345_s1
"""
import argparse
from ultralytics import YOLO

# ---- recipe of the original runs (from their args.yaml) ----
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
ap.add_argument("--model", required=True, help="model yaml (e.g. paper_yolov8s_sgc_p345.yaml)")
ap.add_argument("--weights", default=None, help="COCO weights, loaded where names and shapes match (e.g. yolov8s.pt)")
ap.add_argument("--data", required=True, help="corn data.yaml")
ap.add_argument("--epochs", type=int, default=200)
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--device", default="0")
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--dir", default="runs/seed")
ap.add_argument("--amp", action="store_true",
                help="mixed precision, as in the original runs")
args = ap.parse_args()

model = YOLO(args.model)                 # architecture from the yaml
if args.weights:
    model.load(args.weights)             # COCO weights where names and shapes match

model.train(
    data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
    device=args.device, seed=args.seed, name=args.name, project=args.dir,
    amp=args.amp, workers=8, verbose=True,
    **RECIPE,
)
print("DONE:", args.name)
