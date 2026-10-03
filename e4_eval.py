# -*- coding: utf-8 -*-
"""e4_eval.py - test-set evaluation of the pruned models of Table 7, seeds 0-4.

Seed 0 is the original pruned checkpoint (weights/prune/<cfg>/best.pt); seeds 1-4 are the runs of
run_prune5.sh (runs/prune5/<cfg>_s<k>), scored once run_prune5.sh lists them in
paper_results/prune5_done.txt. Scoring as in table1_eval.py (model.val, test split, 640); parameters
and GFLOPs of the BN-fused model, as in Table 7. Results are cached in paper_results/prune5_results.json.

    python e4_eval.py
"""
import json, os, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = "paper_results/prune5_results.json"
DONE = "paper_results/prune5_done.txt"
CFGS = ["yolov5s", "yolov5s_sgc_p345", "yolov5s_sgc_p45",
        "yolov8s", "yolov8s_sgc_p345", "yolov8s_sgc_p45",
        "yolov9s", "yolov9s_sgc_p345", "yolov9s_sgc_p45",
        "yolov11s", "yolov11s_sgc_p345", "yolov11s_sgc_p45"]
TMP = tempfile.mkdtemp(prefix="e4_val_")


def evaluate(pt):
    from ultralytics import YOLO
    r = YOLO(pt).val(data="ultralytics/cfg/datasets/corn.yaml", split="test", imgsz=640, device="0",
                     verbose=False, plots=False, project=TMP, name="v", exist_ok=True)
    per = {str(r.names[int(c)]): float(r.box.ap50[i]) for i, c in enumerate(r.box.ap_class_index)}
    return {"P": float(r.box.mp), "R": float(r.box.mr), "mAP50": float(r.box.map50),
            "mAP50_95": float(r.box.map), "per": per}


def cost(pt):
    import copy
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops
    m = copy.deepcopy(YOLO(pt).model).float().eval().fuse(verbose=False)
    shapes = [tuple(p.shape) for p in m.parameters()]
    return {"params_M_fused": sum(p.numel() for p in m.parameters()) / 1e6, "GFLOPs_fused": get_flops(m, 640),
            "_shapes": shapes}


def main():
    done = set(open(DONE).read().split()) if os.path.exists(DONE) else set()
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for c in CFGS:
        res.setdefault(c, {})
        ref_shapes = None
        for s in range(5):
            pt = f"weights/prune/{c}/best.pt" if s == 0 else f"runs/prune5/{c}_s{s}/weights/best.pt"
            if s and f"{c}_s{s}" not in done:
                continue
            if not os.path.exists(pt):
                continue
            k = cost(pt)
            shapes = k.pop("_shapes")
            ref_shapes = shapes if s == 0 else ref_shapes
            same = (shapes == ref_shapes) if ref_shapes is not None else None
            if pt not in res[c]:
                res[c][pt] = {**evaluate(pt), **k, "seed": s}
            res[c][pt]["same_arch_as_seed0"] = same
            r = res[c][pt]
            print(f"{c:18s} s{s} mAP50={r['mAP50']:.4f} mAP50_95={r['mAP50_95']:.4f} far={r['per'].get('far', float('nan')):.4f} "
                  f"params={r['params_M_fused']:.3f}M GFLOPs={r['GFLOPs_fused']:.1f} same_arch={same}", flush=True)
            json.dump(res, open(OUT, "w"), indent=1)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
