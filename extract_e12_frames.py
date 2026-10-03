# -*- coding: utf-8 -*-
"""extract_e12_frames.py - external test frames from the Nam Sach (Hai Phong) videos.

Ten videos recorded on 5 July 2026, one per field. N frames are taken from each video, evenly spaced
in time, skipping the first and last 2 s (camera shake when recording starts and stops), so every
field contributes the same number of frames. Each frame is cropped to the central square of the
16:9 frame and resized to 640x640.

    python extract_e12_frames.py --videos videos_namsach
"""
import argparse, csv, glob, os

import cv2

ap = argparse.ArgumentParser()
ap.add_argument("--videos", required=True)
ap.add_argument("--out", default="datasets_e12")
ap.add_argument("--per_video", type=int, default=20)
ap.add_argument("--trim_s", type=float, default=2.0)
ap.add_argument("--size", type=int, default=640)
args = ap.parse_args()

os.makedirs(f"{args.out}/images", exist_ok=True)
rows = []
for field, v in enumerate(sorted(glob.glob(f"{args.videos}/*.mp4")), start=1):
    cap = cv2.VideoCapture(v)
    n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    a, b = int(args.trim_s * fps), n - 1 - int(args.trim_s * fps)
    idx = [round(a + i * (b - a) / (args.per_video - 1)) for i in range(args.per_video)]
    stem = os.path.splitext(os.path.basename(v))[0]
    for k in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        ok, fr = cap.read()
        if not ok:
            print(f"CANNOT READ {stem} frame {k}")
            continue
        h, w = fr.shape[:2]
        s = min(h, w)
        sq = fr[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2]
        sq = cv2.resize(sq, (args.size, args.size), interpolation=cv2.INTER_AREA)
        name = f"ns_f{field:02d}_{stem}_{k:05d}.jpg"
        cv2.imwrite(f"{args.out}/images/{name}", sq, [cv2.IMWRITE_JPEG_QUALITY, 95])
        rows.append({"image": name, "field": field, "video": os.path.basename(v), "frame": k,
                     "time_s": round(k / fps, 2), "src_w": w, "src_h": h, "fps": round(fps, 3)})
    print(f"field {field:2d}  {stem}  {n} frames @ {fps:.2f} fps  -> {sum(r['field'] == field for r in rows)} images, "
          f"spacing {(b - a) / (args.per_video - 1) / fps:.2f} s", flush=True)

with open(f"{args.out}/frames.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)
print(f"-> {len(rows)} images in {args.out}/images, list in {args.out}/frames.csv")
