# -*- coding: utf-8 -*-
"""e12_external_test.py - external test set (Nam Sach): out-of-fold test of the trained models.

Ten fields, one 8K video per field, 20 annotated frames per video (datasets_e12/). The videos form five
folds of consecutive pairs (f01-02, f03-04, f05-06, f07-08, f09-10). In each fold the labels of the pair
(40 frames) set the input scale, and the other eight videos (160 frames) are the test set. The frames are
taken from the 8K videos at imgsz_k = 32*round(640*s_k/32), where s_k is the geometric mean over the four
bands of sqrt(w*h) of the mean training box divided by sqrt(w*h) of the mean box of pair k. Checkpoints
are scored with model.val as in Table 1, and AP is pooled over the out-of-fold test frames of the five
folds; paper_results/e12_external_test.json lists the result of every training run.

    python e12_external_test.py protocol   # input scale of each fold -> paper_results/e12/protocol.json
    python e12_external_test.py prep       # frames from the 8K videos (folder E12_VIDEOS) -> datasets_e12_sq/s{size}/
    python e12_external_test.py zs         # model.val of every checkpoint at each input size
    python e12_external_test.py oof        # pooled out-of-fold AP -> paper_results/e12/oof_summary.json
"""
import csv, glob, json, os, shutil, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
os.environ["YOLO_AUTOINSTALL"] = "false"

E12 = os.path.join(ROOT, "datasets_e12")
SQ = os.path.join(ROOT, "datasets_e12_sq")
OUT = os.path.join(ROOT, "paper_results", "e12")
VIDEOS = os.environ.get("E12_VIDEOS", "videos_namsach")   # folder with the ten 8K videos (only needed for prep)
NAMES = ["near", "mid_near", "mid_far", "far"]
FOLDS = [("f01", "f02"), ("f03", "f04"), ("f05", "f06"), ("f07", "f08"), ("f09", "f10")]


def field(name):
    return name.split("_")[1]


def band_sizes(files):
    rows = []
    for f in files:
        rows += np.loadtxt(f, ndmin=2).tolist()
    a = np.array(rows)
    return np.array([[a[a[:, 0] == c, 3].mean(), a[a[:, 0] == c, 4].mean()] for c in range(4)])


def fold_sizes():
    tr = band_sizes(glob.glob("datasets/train/labels/*.txt"))
    out = []
    for p in FOLDS:
        e = band_sizes([f for f in glob.glob(f"{E12}/labels/*.txt") if field(os.path.basename(f)) in p])
        s = float(np.exp(np.log(np.sqrt(tr[:, 0] * tr[:, 1] / (e[:, 0] * e[:, 1]))).mean()))
        out.append({"pair": list(p), "s": s, "imgsz": int(32 * round(640 * s / 32))})
    return out


def checkpoints():
    t1 = json.load(open("paper_results/table1_results.json"))
    ck = {c: list(v) for c, v in t1.items()}
    return ck


def seed_of(pt):
    if "/main/" in pt:
        return 0
    for s in range(1, 5):
        if f"_s{s}/" in pt or f"_s{s}." in pt:
            return s
    if "_s0/" in pt:
        return 0
    raise ValueError(pt)


# ---------------------------------------------------------------- validator that records per-image statistics
def rec_validator():
    from ultralytics.models.yolo.detect.val import DetectionValidator

    class RecValidator(DetectionValidator):
        REC = []

        def update_metrics(self, preds, batch):
            files, orig, idx = list(batch["im_file"]), self.metrics.update_stats, [0]

            def wrap(stat):
                RecValidator.REC.append((os.path.basename(files[idx[0]]),
                                         {k: np.array(stat[k]).copy() for k in ("tp", "conf", "pred_cls", "target_cls")}))
                idx[0] += 1
                return orig(stat)

            self.metrics.update_stats = wrap
            try:
                super().update_metrics(preds, batch)
            finally:
                self.metrics.update_stats = orig

    return RecValidator


def run_val(pt, yaml_path, imgsz, out_npz, tmp):
    from ultralytics import YOLO
    V = rec_validator()
    V.REC.clear()
    r = YOLO(pt).val(data=yaml_path, split="test", imgsz=imgsz, device="0", verbose=False, plots=False,
                     validator=V, project=tmp, name="v", exist_ok=True)
    names = [n for n, _ in V.REC]
    tp = np.concatenate([d["tp"] for _, d in V.REC]).astype(bool)
    conf = np.concatenate([d["conf"] for _, d in V.REC])
    pcls = np.concatenate([d["pred_cls"] for _, d in V.REC])
    tcls = np.concatenate([d["target_cls"] for _, d in V.REC])
    pimg = np.concatenate([np.full(len(d["conf"]), i) for i, (_, d) in enumerate(V.REC)])
    timg = np.concatenate([np.full(len(d["target_cls"]), i) for i, (_, d) in enumerate(V.REC)])
    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    np.savez_compressed(out_npz, names=np.array(names), tp=tp, conf=conf, pred_cls=pcls, target_cls=tcls,
                        pred_img=pimg, target_img=timg)
    ci = list(r.box.ap_class_index)
    return {"mAP50": float(r.box.map50), "mAP50_95": float(r.box.map), "far": float(r.box.ap50[ci.index(3)]),
            "n_images": len(names)}


def write_yaml(path, root, train="images", val="images", test="images"):
    with open(path, "w") as f:
        f.write(f"path: {root}\ntrain: {train}\nval: {val}\ntest: {test}\nnames:\n"
                + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)))


# ---------------------------------------------------------------- steps
def cmd_protocol():
    p = os.path.join(OUT, "protocol.json")
    if os.path.exists(p):
        sys.exit(f"{p} exists and is not overwritten.")
    os.makedirs(OUT, exist_ok=True)
    proto = {"created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"), "folds": fold_sizes(),
             "metrics": ["mAP50", "far AP50", "mAP50-95"],
             "aggregation": "AP pooled over the out-of-fold predictions of the five folds; mean +/- SD over five seeds",
             "note": "ten fields, one video per field; each fold uses one pair of consecutive videos for calibration"}
    json.dump(proto, open(p, "w"), indent=1)
    print(json.dumps(proto["folds"], indent=1))


def sizes_all():
    proto = json.load(open(os.path.join(OUT, "protocol.json")))
    return sorted({640} | {f["imgsz"] for f in proto["folds"]}), proto


def cmd_prep():
    import cv2
    sizes, _ = sizes_all()
    rows = list(csv.DictReader(open(f"{E12}/frames.csv")))
    for s in sizes:
        os.makedirs(f"{SQ}/s{s}/images", exist_ok=True)
        if os.path.isdir(f"{SQ}/s{s}/labels"):
            shutil.rmtree(f"{SQ}/s{s}/labels")
        shutil.copytree(f"{E12}/labels", f"{SQ}/s{s}/labels")
        write_yaml(f"{SQ}/s{s}/data.yaml", f"{SQ}/s{s}")
    caps = {}
    for i, r in enumerate(sorted(rows, key=lambda r: (r["video"], int(r["frame"])))):
        if all(os.path.exists(f"{SQ}/s{s}/images/{r['image']}") for s in sizes):
            continue
        cap = caps.setdefault(r["video"], cv2.VideoCapture(f"{VIDEOS}/{r['video']}"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(r["frame"]))
        ok, fr = cap.read()
        assert ok, r
        h, w = fr.shape[:2]
        q = min(h, w)
        sq = fr[(h - q) // 2:(h + q) // 2, (w - q) // 2:(w + q) // 2]
        for s in sizes:
            cv2.imwrite(f"{SQ}/s{s}/images/{r['image']}", cv2.resize(sq, (s, s), interpolation=cv2.INTER_AREA),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
        if i % 20 == 0:
            print("prep", i, r["image"], flush=True)
    # check: s640 must equal the annotated frames (same extraction procedure)
    d = [np.abs(cv2.imread(f"{SQ}/s640/images/{n}").astype(float) - cv2.imread(f"{E12}/images/{n}").astype(float)).mean()
         for n in sorted(os.listdir(f"{E12}/images"))[::20]]
    print("MAD of s640 against datasets_e12/images (10 images): max %.2f" % max(d))


def cmd_zs():
    import tempfile
    sizes, _ = sizes_all()
    tmp = tempfile.mkdtemp(prefix="e12_")
    summ_p = os.path.join(OUT, "zs_summary.json")
    summ = json.load(open(summ_p)) if os.path.exists(summ_p) else {}
    for cfg, pts in checkpoints().items():
        for pt in pts:
            for s in sizes:
                key = f"{cfg}|{seed_of(pt)}|{s}"
                npz = os.path.join(OUT, "zs", f"{cfg}__s{seed_of(pt)}__{s}.npz")
                if key in summ and os.path.exists(npz):
                    continue
                summ[key] = {"pt": pt, **run_val(pt, f"{SQ}/s{s}/data.yaml", s, npz, tmp)}
                json.dump(summ, open(summ_p, "w"), indent=1)
                print("zs", key, "mAP50 %.4f far %.4f" % (summ[key]["mAP50"], summ[key]["far"]), flush=True)


def _ap(parts):
    """Pooled AP from a list of (tp, conf, pred_cls, target_cls), as Ultralytics ap_per_class."""
    from ultralytics.utils.metrics import ap_per_class
    tp, conf, pc, tc = [np.concatenate([p[i] for p in parts]) for i in range(4)]
    r = ap_per_class(tp, conf, pc, tc, plot=False)
    ap, uc = r[5], list(r[6])
    return {"mAP50": float(ap[:, 0].mean()), "far": float(ap[uc.index(3), 0]) if 3 in uc else float("nan"),
            "mAP50_95": float(ap.mean())}


def _subset(npz, keep):
    """Statistics of the images whose name satisfies keep(name), from a per-image npz file."""
    z = np.load(npz)
    names = [str(n) for n in z["names"]]
    ok = np.array([keep(n) for n in names])
    pm, tm = ok[z["pred_img"]], ok[z["target_img"]]
    return (z["tp"][pm], z["conf"][pm], z["pred_cls"][pm], z["target_cls"][tm]), int(ok.sum())


def cmd_oof():
    """Pooled out-of-fold AP for every checkpoint -> paper_results/e12/oof_summary.json."""
    _, proto = sizes_all()
    folds = [(k, tuple(f["pair"]), f["imgsz"]) for k, f in enumerate(proto["folds"])]
    out = {"unchanged": {}, "scale": {}, "per_fold": {}, "n_images_per_fold": {}}
    for cfg, pts in checkpoints().items():
        for pt in pts:
            sd = str(seed_of(pt))
            for cond in ("unchanged", "scale"):
                parts = []
                for k, pair, size in folds:
                    s = 640 if cond == "unchanged" else size
                    p, n = _subset(os.path.join(OUT, "zs", f"{cfg}__s{sd}__{s}.npz"), lambda x, pr=pair: field(x) not in pr)
                    parts.append(p)
                    out["per_fold"].setdefault(cond, {}).setdefault(cfg, {}).setdefault(sd, {})[f"fold{k}"] = _ap([p])
                    out["n_images_per_fold"][f"fold{k}"] = n
                out[cond].setdefault(cfg, {})[sd] = _ap(parts)
    out["protocol"] = proto
    json.dump(out, open(os.path.join(OUT, "oof_summary.json"), "w"), indent=1)
    for cond in ("unchanged", "scale"):
        for cfg, d in out[cond].items():
            v = np.array([[x["mAP50"], x["far"]] for x in d.values()]) * 100
            print(f"{cond:9s} {cfg:18s} mAP50 {v[:, 0].mean():5.1f}+/-{v[:, 0].std(ddof=1):4.1f}  far {v[:, 1].mean():5.1f}+/-{v[:, 1].std(ddof=1):4.1f}")


if __name__ == "__main__":
    {"protocol": cmd_protocol, "prep": cmd_prep, "zs": cmd_zs, "oof": cmd_oof}[sys.argv[1]]()
