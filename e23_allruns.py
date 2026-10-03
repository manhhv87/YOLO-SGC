# -*- coding: utf-8 -*-
"""e23_allruns.py - inference-time interventions and prior-branch deletion on all 40 trained SGC models.

Interventions: the inference-time interventions of Table 4 on 8 SGC configurations x 5 seeds, using
    scalemap_intervention.py:
      scale map : zero, const, vflip, shuffle, ramp   (upper block of Table 4)
      fusion    : uniform, zmean, zshuffle            (lower block of Table 4)
    Each (checkpoint, target) writes one JSON file to paper_results/interv_allruns/.
Deployment (Table 6): the prior branch is deleted with make_deploy_generic.py on the same 40
    checkpoints and the test set is scored as in table1_eval.py -> paper_results/deploy_allruns.json.

Checkpoint paths are those of table1_eval.py. The script can be re-run: finished parts are skipped.
    python e23_allruns.py
"""
import json, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
PY = sys.executable

BB = ["yolov5s", "yolov8s", "yolov9s", "yolov11s"]
VAR = ["_sgc_p345", "_sgc_p45"]
SCALEMAP_MODES = "none,zero,const,vflip,shuffle,ramp"
FUSION_MODES = "none,uniform,zmean,zshuffle"
INTERV_DIR = "paper_results/interv_allruns"
DEPLOY_DIR = "weights/deploy_allruns"
DEPLOY_OUT = "paper_results/deploy_allruns.json"
LOG = "paper_results/e23_progress.log"
RUNLOG = "paper_results/e23_run.log"


def paths(b, v):
    """Five checkpoints, seeds 0..4 (as in table1_eval.py)."""
    p = [f"weights/main/{b}{v}/best.pt"]
    if b == "yolov8s":
        p += [f"weights/seed/{b}{v}_s{s}/best.pt" for s in (1, 2)]
        p += [f"runs/seed5/{b}{v}_s{s}/weights/best.pt" for s in (3, 4)]
    else:
        p += [f"runs/table1/{b}{v}_s{s}/weights/best.pt" for s in (1, 2, 3, 4)]
    return p


def log(msg):
    with open(LOG, "a") as f:
        f.write(f"{time.strftime('%F %T')} {msg}\n")


def call(cmd):
    with open(RUNLOG, "a") as f:
        f.write(f"\n$ {' '.join(cmd)}\n")
        f.flush()
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return r.returncode


def evaluate(pt):
    from ultralytics import YOLO
    r = YOLO(pt).val(data="ultralytics/cfg/datasets/corn.yaml", split="test", imgsz=640,
                     device="0", verbose=False, plots=False)
    per = {str(r.names[int(c)]): float(r.box.ap50[i]) for i, c in enumerate(r.box.ap_class_index)}
    return {"P": float(r.box.mp), "R": float(r.box.mr), "mAP50": float(r.box.map50),
            "mAP50_95": float(r.box.map), "per": per}


def main():
    os.makedirs(INTERV_DIR, exist_ok=True)
    os.makedirs(DEPLOY_DIR, exist_ok=True)
    runs = [(b + v, s, pt) for b in BB for v in VAR for s, pt in enumerate(paths(b, v))]
    log(f"START: {len(runs)} checkpoints")

    # inference-time interventions
    for cfg, s, pt in runs:
        for target, modes in (("scalemap", SCALEMAP_MODES), ("fusion", FUSION_MODES)):
            out = f"{INTERV_DIR}/{cfg}_s{s}_{target}.json"
            if os.path.exists(out):
                continue
            log(f"intervention {cfg} s{s} {target}")
            rc = call([PY, "scalemap_intervention.py", "--ckpt", pt, "--target", target,
                       "--modes", modes, "--out", out])
            if rc:
                log(f"  ERROR intervention {cfg} s{s} {target} (rc={rc})")
    log("interventions done")

    # delete the prior branch, then score the test set
    res = json.load(open(DEPLOY_OUT)) if os.path.exists(DEPLOY_OUT) else {}
    for cfg, s, pt in runs:
        key = f"{cfg}_s{s}"
        if key in res:
            continue
        dst = f"{DEPLOY_DIR}/{key}_deploy"
        if not os.path.exists(dst + ".pt"):
            log(f"deploy {key}")
            rc = call([PY, "make_deploy_generic.py", "--src", pt, "--out", dst,
                       "--name", f"e3_{cfg}_deploy"])
            if rc or not os.path.exists(dst + ".pt"):
                log(f"  ERROR deploy {key} (rc={rc})")
                continue
        res[key] = {"src": pt, "deploy": dst + ".pt", **evaluate(dst + ".pt")}
        json.dump(res, open(DEPLOY_OUT, "w"), indent=1)
        log(f"deployed {key} mAP50={res[key]['mAP50']:.4f} far={res[key]['per'].get('far', float('nan')):.4f}")
    log("deployment done; all finished")


if __name__ == "__main__":
    main()
