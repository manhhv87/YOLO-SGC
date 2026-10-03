# Jetson Nano: TensorRT FP16

Scripts behind the Nano FP16 columns of Tables 1 and 7. Device: Jetson Nano 4 GB, JetPack 4.6.6
(L4T R32.7.6), TensorRT 8.2.1, power mode MAXN with `jetson_clocks`.

| File | Runs on | Purpose |
|---|---|---|
| `export_e5.py` | PC | exports the seed-0 checkpoints to ONNX (batch 1, 640x640, FP32, opset 12, no NMS) |
| `trt_common.py` | Nano and PC | pre-processing and NMS used on both sides |
| `nano_bench.sh` | Nano | builds the FP16 engines with `trtexec` and times them |
| `nano_eval.py` | Nano | runs every engine on the 500 test images and stores the predictions |
| `score_e5.py` | PC | reads the timing, scores the FP16 predictions and the FP32 ONNX reference, writes `paper_results/e5_trt.json` |

On the PC (`export_e5.py` needs `onnx`, `onnxslim` and `onnxruntime`):

```bash
python e5_nano/export_e5.py                    # -> e5_nano/onnx/
mkdir -p e5_nano/test_images && cp datasets/test/images/* e5_nano/test_images/
```

Copy `e5_nano/` to the Nano. There, with the system Python 3.6 of JetPack (no PyTorch needed):

```bash
sudo nvpmodel -m 0 && sudo jetson_clocks
cd e5_nano && mkdir -p results_nano
./nano_bench.sh 2>&1 | tee results_nano/bench.out       # FP16 engines and timing (several hours)
python3 nano_eval.py 2>&1 | tee results_nano/eval.out    # FP16 predictions on the test images
```

Copy `results_nano/` back to `e5_nano/results_nano/` on the PC and score:

```bash
python e5_nano/score_e5.py --ref               # FP32 reference: ONNX Runtime on the CPU
python e5_nano/score_e5.py                     # -> paper_results/e5_trt.json
```

FP16 throughput is 1000 divided by the mean GPU compute time reported by `trtexec` (batch 1). The
accuracy change is measured between the FP16 engine and the FP32 ONNX model with the same 640x640
input and the same pre- and post-processing.
