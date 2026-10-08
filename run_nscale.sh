#!/bin/bash
# n-scale references: YOLOv8n and YOLO11n, seeds 0-4 (10 runs).
# train_seed.py with --amp, initialized from the COCO weights.
cd "$(dirname "$0")" || exit 1
PY=${PY:-python}
DATA=ultralytics/cfg/datasets/corn.yaml
LOG=paper_results/nscale_progress.log
DONE=paper_results/nscale_done.txt
touch "$DONE"
# run name -> (model yaml, COCO weights)
declare -A Y=( [yolov8n]=yolov8n.yaml [yolov11n]=yolo11n.yaml )
declare -A W=( [yolov8n]=yolov8n.pt [yolov11n]=yolo11n.pt )
echo "START $(date '+%F %T')" >> "$LOG"
for m in yolov8n yolov11n; do
  for s in 0 1 2 3 4; do
    name="${m}_s${s}"
    grep -qx "$name" "$DONE" && { echo "SKIP $name" >> "$LOG"; continue; }
    rm -rf "runs/nscale/${name}"
    echo "RUN $name $(date '+%F %T')" >> "$LOG"
    $PY train_seed.py --model "${Y[$m]}" --weights "${W[$m]}" \
        --data "$DATA" --seed "$s" --name "$name" --dir runs/nscale --amp \
        >> paper_results/nscale_train.log 2>&1
    [ -f "runs/nscale/${name}/weights/best.pt" ] && echo "$name" >> "$DONE"
    echo "DONE $name $(date '+%F %T')" >> "$LOG"
  done
done
echo "NSCALE_DONE $(date '+%F %T')" >> "$LOG"
