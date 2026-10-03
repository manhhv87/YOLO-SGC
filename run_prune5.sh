#!/bin/bash
# Seeds 1-4 of the pruned models of Table 7 (seed 0 = weights/prune/<cfg>/best.pt).
# 12 configurations x 4 seeds = 48 runs, each with the pruned architecture of seed 0 (--ref).
cd "$(dirname "$0")" || exit 1
PY=${PY:-python}
DATA=ultralytics/cfg/datasets/corn.yaml
LOG=paper_results/prune5_progress.log
DONE=paper_results/prune5_done.txt
touch "$DONE"
# configuration -> (yaml of the architecture before pruning, COCO weights used for seed 0)
declare -A Y=( [yolov8s_sgc_p45]=e4_yolov8n_sgc_p45_prune.yaml [yolov8s_sgc_p345]=e4_yolov8s_sgc_p345_prune.yaml
               [yolov8s]=e4_yolov8s_prune.yaml [yolov5s_sgc_p345]=e4_yolov5s_sgc_p345_prune.yaml
               [yolov5s_sgc_p45]=e4_yolov5s_sgc_p45_prune.yaml [yolov5s]=e4_yolov5s_prune.yaml
               [yolov11s]=e4_yolov11s_prune.yaml [yolov11s_sgc_p345]=e4_yolov11s_sgc_p345_prune.yaml
               [yolov11s_sgc_p45]=e4_yolov11s_sgc_p45_prune.yaml [yolov9s]=e4_yolov9s_prune.yaml
               [yolov9s_sgc_p345]=e4_yolov9s_sgc_p345_prune.yaml [yolov9s_sgc_p45]=e4_yolov9s_sgc_p45_prune.yaml )
declare -A W=( [yolov8s_sgc_p45]=yolov8s.pt [yolov8s_sgc_p345]=yolov8s.pt [yolov8s]=yolov8s.pt
               [yolov5s_sgc_p345]=yolov5su.pt [yolov5s_sgc_p45]=yolov5su.pt [yolov5s]=yolov5su.pt
               [yolov11s]=yolo11s.pt [yolov11s_sgc_p345]=yolo11s.pt [yolov11s_sgc_p45]=yolo11s.pt
               [yolov9s]=yolov9s.pt [yolov9s_sgc_p345]=yolov9s.pt [yolov9s_sgc_p45]=yolov9s.pt )
echo "START $(date '+%F %T')" >> "$LOG"
for c in yolov8s_sgc_p45 yolov8s_sgc_p345 yolov8s yolov5s_sgc_p345 yolov5s_sgc_p45 yolov5s \
         yolov11s yolov11s_sgc_p345 yolov11s_sgc_p45 yolov9s yolov9s_sgc_p345 yolov9s_sgc_p45; do
  for s in 1 2 3 4; do
    name="${c}_s${s}"
    grep -qx "$name" "$DONE" && { echo "SKIP $name" >> "$LOG"; continue; }
    rm -rf "runs/prune5/${name}"
    echo "RUN $name $(date '+%F %T')" >> "$LOG"
    $PY train_seed_prune.py --yaml "${Y[$c]}" --weights "${W[$c]}" --ref "weights/prune/${c}/best.pt" \
        --data "$DATA" --seed "$s" --name "$name" --dir runs/prune5 --amp \
        >> paper_results/prune5_train.log 2>&1
    [ -f "runs/prune5/${name}/weights/best.pt" ] && echo "$name" >> "$DONE"
    echo "DONE $name $(date '+%F %T')" >> "$LOG"
  done
done
echo "PRUNE5_DONE $(date '+%F %T')" >> "$LOG"
