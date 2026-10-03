#!/bin/bash
# nano_bench.sh - on the Jetson Nano: build TensorRT engines from onnx/*.onnx with trtexec and time them.
#
#   sudo nvpmodel -m 0 && sudo jetson_clocks       # the power mode used for the FP32 measurements
#   ./nano_bench.sh                                  # all 26 models, FP16
#   ./nano_bench.sh onnx/yolov8s_sgc_p45_pruned.onnx # a single model
#
# For each model:
#   (1) build the engine if it does not exist; log in results_nano/<name>_<prec>_build.log
#   (2) time the saved engine separately: 2 s warm-up, at least DURATION seconds, --useSpinWait;
#       per-inference times go to results_nano/<name>_<prec>_times.json (percentiles computed on the PC).
# Device state (nvpmodel, L4T, TensorRT version, temperature) is recorded at the start and the end.
set -u
cd "$(dirname "$0")"
TRTEXEC=${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}
PRECISIONS=${PRECISIONS:-fp16}
DURATION=${DURATION:-30}
[ -x "$TRTEXEC" ] || { echo "trtexec not found at $TRTEXEC (set TRTEXEC=...)"; exit 1; }
mkdir -p engines results_nano

state() {
  echo "### $(date '+%F %T')"
  echo "--- nvpmodel";      nvpmodel -q 2>&1 | head -5
  echo "--- jetson_clocks"; (sudo -n jetson_clocks --show 2>/dev/null || echo "(sudo needed)") | head -12
  echo "--- L4T";           head -1 /etc/nv_tegra_release 2>/dev/null
  echo "--- TensorRT";      dpkg -l 2>/dev/null | grep -E 'libnvinfer8 |libnvinfer7 |tensorrt ' | awk '{print $2, $3}'
  echo "--- tegrastats";    timeout 3 tegrastats --interval 1000 2>/dev/null | head -1
  echo "--- free -m";       free -m
}
state > results_nano/device_state_start.txt
cat results_nano/device_state_start.txt

MODELS=("$@"); [ ${#MODELS[@]} -eq 0 ] && MODELS=(onnx/*.onnx)
for onnx in "${MODELS[@]}"; do
  name=$(basename "$onnx" .onnx)
  for prec in $PRECISIONS; do
    eng="engines/${name}_${prec}.engine"; flag=""; [ "$prec" = fp16 ] && flag="--fp16"
    if [ ! -s "$eng" ]; then
      echo "[$(date +%T)] build $name $prec ..."
      if ! "$TRTEXEC" --onnx="$onnx" $flag --workspace=1024 --saveEngine="$eng" \
           > "results_nano/${name}_${prec}_build.log" 2>&1; then
        echo "  !! build failed, see results_nano/${name}_${prec}_build.log"; rm -f "$eng"; continue
      fi
    fi
    echo "[$(date +%T)] time  $name $prec ..."
    if ! "$TRTEXEC" --loadEngine="$eng" --warmUp=2000 --duration="$DURATION" --iterations=100 --avgRuns=100 \
         --useSpinWait --exportTimes="results_nano/${name}_${prec}_times.json" \
         > "results_nano/${name}_${prec}.log" 2>&1; then
      echo "  !! timing failed, see results_nano/${name}_${prec}.log"; continue
    fi
    gpu=$(grep 'GPU Compute Time: min' "results_nano/${name}_${prec}.log" | tail -1 | sed -n 's/.*mean = \([0-9.]*\) ms.*/\1/p')
    temp=$(timeout 3 tegrastats --interval 1000 2>/dev/null | head -1 | grep -o 'GPU@[0-9.]*C')
    echo "  GPU compute mean = ${gpu} ms  (${temp})"
    printf '%s\t%s\t%s\t%s\n' "$name" "$prec" "$gpu" "$temp" >> results_nano/summary.tsv
  done
done
state > results_nano/device_state_end.txt
echo "Timing done. Next: python3 nano_eval.py (accuracy), then copy results_nano/ to the PC."
