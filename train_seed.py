# -*- coding: utf-8 -*-
"""
train_seed.py — Train dung DUNG recipe goc (trich tu args.yaml cua cac run corn)
de chay multi-seed. Dung API framework (YOLO) + partial-load COCO nhu train.py --weights.

Vi du (train YOLOv8s-SGC-P345, seed 1, khoi tao COCO):
    python train_seed.py --model yolov8s_sgc_p345.yaml --weights yolov8s.pt \
        --data ~/datasets/corn/data.yaml --seed 1 --name yolov8s_sgc_p345_s1

Chay het multi-seed: xem vong lap o cuoi file (bo comment) hoac dung bash loop.
"""
import argparse
from ultralytics import YOLO

# ---- RECIPE GOC: copy nguyen tu args.yaml cua run corn (KHONG dung mac dinh Ultralytics) ----
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
ap.add_argument("--model", required=True, help="*.yaml kien truc (vd yolov8s_sgc_p345.yaml)")
ap.add_argument("--weights", default=None, help="*.pt COCO de partial-load vao backbone (vd yolov8s.pt)")
ap.add_argument("--data", required=True, help="corn data.yaml")
ap.add_argument("--epochs", type=int, default=200)
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--device", default="0")
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--dir", default="runs/seed")
ap.add_argument("--amp", action="store_true",
                help="Bat AMP (run goc dung amp=true). Neu mAP sup ve ~0 thi BO co nay.")
args = ap.parse_args()

model = YOLO(args.model)                 # dung kien truc tu YAML (co AMRF/SGCBlock)
if args.weights:
    model.load(args.weights)             # partial-load COCO (chi cac layer khop key+shape)

model.train(
    data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
    device=args.device, seed=args.seed, name=args.name, project=args.dir,
    amp=args.amp, workers=8, verbose=True,
    **RECIPE,
)
print("DONE:", args.name)
