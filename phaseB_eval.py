# -*- coding: utf-8 -*-
"""phaseB_eval.py — evaluate the four Phase B arms on the test set and run the pre-registered tests.

Protocol: primary endpoint far AP@50, secondary mAP@50; n=5 seeds per arm; Welch t-test
and Mann-Whitney; threshold p<0.05; smallest detectable difference about 1.6 mAP points.
"""
import glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BRANCHES = {
    "Baseline": ["weights/main/yolov8s/best.pt",
                 "weights/seed/yolov8s_s1/best.pt",
                 "weights/seed/yolov8s_s2/best.pt",
                 "runs/seed5/yolov8s_s3/weights/best.pt",
                 "runs/seed5/yolov8s_s4/weights/best.pt"],
    "Full": ["weights/main/yolov8s_sgc_p345/best.pt",
             "weights/seed/yolov8s_sgc_p345_s1/best.pt",
             "weights/seed/yolov8s_sgc_p345_s2/best.pt",
             "runs/seed5/yolov8s_sgc_p345_s3/weights/best.pt",
             "runs/seed5/yolov8s_sgc_p345_s4/weights/best.pt"],
    "NoScale": ["weights/ablation/NoScale/best.pt"] +
               [f"runs/phaseB/noscale_s{s}/weights/best.pt" for s in range(1, 5)],
    "ShufPrior": [f"runs/phaseB/shufprior_s{s}/weights/best.pt" for s in range(5)],
    "YPrior": [f"runs/ablation2/yolov8s_sgc_p345_yprior_s{s}/weights/best.pt" for s in range(5)],
    "AMRF_FPN": ["weights/ablation/AMRF_FPN/best.pt"] +
                [f"runs/ablation5/AMRF_FPN_s{s}/weights/best.pt" for s in range(1, 5)],
    "EqualWeight": ["weights/ablation/EqualWeight/best.pt"] +
                   [f"runs/ablation5/EqualWeight_s{s}/weights/best.pt" for s in range(1, 5)],
    "NoAMRF": ["weights/ablation/NoAMRF/best.pt"] +
              [f"runs/ablation5/NoAMRF_s{s}/weights/best.pt" for s in range(1, 5)],
    "Minimal": ["weights/ablation/minimal/best.pt"] +
               [f"runs/ablation5/minimal_s{s}/weights/best.pt" for s in range(1, 5)],
}
OUT = "paper_results/phaseB_results.json"


def evaluate(pt):
    from ultralytics import YOLO
    r = YOLO(pt).val(data="ultralytics/cfg/datasets/corn.yaml", split="test", imgsz=640,
                     device="0", verbose=False, plots=False)
    per = {str(r.names[int(c)]): float(r.box.ap50[i]) for i, c in enumerate(r.box.ap_class_index)}
    return {"mAP50": float(r.box.map50), "mAP50_95": float(r.box.map), "per": per}


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for br, paths in BRANCHES.items():
        res.setdefault(br, {})
        for p in paths:
            if not os.path.exists(p):
                print(f"THIẾU {p}")
                continue
            if p in res[br]:
                continue
            res[br][p] = evaluate(p)
            r = res[br][p]
            print(f"{br:10s} {p:52s} mAP50={r['mAP50']:.4f} far={r['per'].get('far', float('nan')):.4f}")
            json.dump(res, open(OUT, "w"), indent=1)
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    os.makedirs("paper_results", exist_ok=True)
    main()
