# -*- coding: utf-8 -*-
"""nscale_eval.py - test-set accuracy of the n-scale references YOLOv8n and YOLO11n, seeds 0-4 (runs/nscale).

Scored as in table1_eval.py; parameters and GFLOPs are counted on the BN-fused model, as in Table 1.
Results are cached in the JSON file, so an interrupted run can be resumed.

    python nscale_eval.py
"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = "paper_results/nscale_results.json"
CFGS = ["yolov8n", "yolov11n"]


def evaluate(pt):
    from ultralytics import YOLO
    r = YOLO(pt).val(data="ultralytics/cfg/datasets/corn.yaml", split="test", imgsz=640,
                     device="0", verbose=False, plots=False)
    per = {str(r.names[int(c)]): float(r.box.ap50[i]) for i, c in enumerate(r.box.ap_class_index)}
    return {"P": float(r.box.mp), "R": float(r.box.mr), "mAP50": float(r.box.map50),
            "mAP50_95": float(r.box.map), "per": per}


def cost(pt):
    import copy
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops
    m = copy.deepcopy(YOLO(pt).model).float().eval().fuse(verbose=False)
    return {"params_M_fused": sum(p.numel() for p in m.parameters()) / 1e6, "GFLOPs_fused": get_flops(m, 640)}


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for c in CFGS:
        res.setdefault(c, {})
        for s in range(5):
            pt = f"runs/nscale/{c}_s{s}/weights/best.pt"
            if pt in res[c] or not os.path.exists(pt):
                continue
            res[c][pt] = {**evaluate(pt), **cost(pt)}
            r = res[c][pt]
            print(f"{c:9s} s{s} mAP50={r['mAP50']:.4f} mAP50_95={r['mAP50_95']:.4f} far={r['per'].get('far', float('nan')):.4f} "
                  f"params={r['params_M_fused']:.2f}M GFLOPs={r['GFLOPs_fused']:.1f}", flush=True)
            json.dump(res, open(OUT, "w"), indent=1)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
