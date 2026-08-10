# -*- coding: utf-8 -*-
"""table1_eval.py — evaluate 12 configurations over 5 seeds for the pre-pruning table.

Seed 0 comes from the original checkpoint in weights/main; seeds 1-4 from runs/table1
(v5s/v9s/v11s) or weights/seed and runs/seed5 (the pre-existing YOLOv8s family).
Results are cached to JSON as they are produced, so re-running is cheap.
"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = "paper_results/table1_results.json"
BB = ["yolov5s", "yolov8s", "yolov9s", "yolov11s"]
VAR = ["", "_sgc_p345", "_sgc_p45"]


def paths(b, v):
    """Five checkpoint paths, seeds 0..4, for one configuration."""
    p = [f"weights/main/{b}{v}/best.pt"]
    if b == "yolov8s":
        p += [f"weights/seed/{b}{v}_s{s}/best.pt" for s in (1, 2)]
        p += [f"runs/seed5/{b}{v}_s{s}/weights/best.pt" for s in (3, 4)]
    else:
        p += [f"runs/table1/{b}{v}_s{s}/weights/best.pt" for s in (1, 2, 3, 4)]
    return p


def evaluate(pt):
    from ultralytics import YOLO
    r = YOLO(pt).val(data="ultralytics/cfg/datasets/corn.yaml", split="test", imgsz=640,
                     device="0", verbose=False, plots=False)
    per = {str(r.names[int(c)]): float(r.box.ap50[i]) for i, c in enumerate(r.box.ap_class_index)}
    return {"P": float(r.box.mp), "R": float(r.box.mr), "mAP50": float(r.box.map50),
            "mAP50_95": float(r.box.map), "per": per}


def main():
    os.makedirs("paper_results", exist_ok=True)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for b in BB:
        for v in VAR:
            key = b + v
            res.setdefault(key, {})
            for pt in paths(b, v):
                if pt in res[key]:
                    continue
                if not os.path.exists(pt):
                    print(f"THIẾU {pt}", flush=True)
                    continue
                res[key][pt] = evaluate(pt)
                r = res[key][pt]
                print(f"{key:22s} {os.path.basename(os.path.dirname(pt)):28s} "
                      f"mAP50={r['mAP50']:.4f} far={r['per'].get('far', float('nan')):.4f}", flush=True)
                json.dump(res, open(OUT, "w"), indent=1)
    print(f"-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
