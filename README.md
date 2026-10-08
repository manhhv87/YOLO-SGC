# YOLO-SGC: Scale-Guided Context Fusion Neck

Reference implementation for the paper

> A Lightweight Scale-Guided Context Fusion Neck for Far-Field Crop-Row Detection on Edge Hardware

The SGC neck replaces the FPN/PAN neck of YOLO detectors. Its three parts are in
`ultralytics/nn/modules/block.py`: an AMRF block that aggregates multi-receptive-field
context through dilated depthwise branches, a ScaleMapHead that derives a spatial prior
from early backbone features without depth supervision, and an SGCBlock that fuses
pyramid levels under spatially varying, feature-conditioned weights. Two configurations:
SGC-P345 (three detection levels) and SGC-P45 (drops the P3 branch, used for
deployment).

## Install

```bash
git clone https://github.com/manhhv87/YOLO-SGC.git
cd YOLO-SGC
pip install -r requirements.txt
```

Results were produced with PyTorch 2.11.0 / CUDA 13.0 on an RTX 5060 Ti, and edge
throughput on an NVIDIA Jetson Nano 4 GB.

## Data and models

The crop-row dataset (2500 images, four distance bands) and the trained checkpoints are
not in this repository; they are available from the corresponding author on reasonable
request. The dataset is expected at `datasets/{train,valid,test}/{images,labels}` in YOLO
format with classes `near`, `mid_near`, `mid_far`, `far`, described by
`ultralytics/cfg/datasets/corn.yaml`.

Model configurations are `ultralytics/cfg/models/v8/paper_*.yaml`. Ablation variants carry
the suffixes `_noscale`, `_AMRF_FPN`, `_yprior` and `_deploy`.

## Reproducing

```bash
python train_seed.py --model paper_yolov8s_sgc_p345.yaml --weights yolov8s.pt \
       --data ultralytics/cfg/datasets/corn.yaml --seed 0
python table1_eval.py                        # detection performance, four backbones
python phaseB_eval.py                        # neck comparison and module ablation
python scalemap_intervention.py              # inference-time interventions
python make_deploy_generic.py                # remove the prior branch after training
bash run_prune5.sh                           # pruned models (see below)
python eval_guidance_line.py --split test    # guidance-line quality
python bench_all_fps.py                      # throughput
```

Training uses SGD (momentum 0.937, weight decay 5e-4), batch 32, cosine schedule with
initial LR 0.005, 640x640 input, at most 200 epochs with patience 30. Accuracy comparisons
are means over five seeds (0-4). `paper_results/*.json` holds the per-checkpoint accuracy
metrics behind the tables.

## Additional experiments

Scripts and results added with the revised manuscript. All results are in `paper_results/`.

| Scripts | Results | Content |
|---|---|---|
| `e23_allruns.py` | `interv_allruns/`, `deploy_allruns.json` | interventions and removal of the prior branch on all 40 trained SGC models (Sections 4.4 and 4.6) |
| `run_prune5.sh`, `train_seed_prune.py`, `e4_eval.py` | `prune5_results.json` | pruned models of Table 7, seeds 0-4 |
| `e5_nano/` | `e5_trt.json` | TensorRT FP16 on the Jetson Nano (Tables 1 and 7; see `e5_nano/README.md`) |
| `extract_e12_frames.py`, `e12_external_test.py` | `e12_external_test.json` | external test set (below) |
| `eval_guidance_seeds.py`, `eval_guidance_bands.py` | `guidance_seeds.json`, `guidance_bands.json`, `guidance_cache/` | guidance line over five runs per configuration (Table 8) and its error by distance band (Section 4.9); `guidance_cache/` holds the highest-confidence box of each class per test image and checkpoint, from which `eval_guidance_bands.py` reruns without the checkpoints |
| `run_nscale.sh`, `nscale_eval.py`, `e5_nano/benchmark_pytorch_e5.py` | `nscale_results.json`, `e5_pytorch_fp32_nscale.csv` | n-scale references YOLOv8n and YOLO11n, seeds 0-4, and their FP32 throughput on the Jetson Nano |

Seed 0 of each pruned model is the original pruned checkpoint; seeds 1-4 are trained by
`run_prune5.sh` with the same per-layer widths. The architectures before pruning are in
`ultralytics/cfg/models/v8/e4_*_prune.yaml`.

### External test set (Nam Sach)

`datasets_e12/` holds 200 annotated frames from ten corn fields in Nam Sach (Hai Phong), recorded
on 5 July 2026, away from the training site and in a different year and growing season. Each field
was recorded in one 8K video by a smartphone (Samsung Galaxy S23) mounted on the robot. Twenty
frames per video, evenly spaced in time, are given as the central square of the frame resized to
640x640 (`images/`), with labels of the four distance bands in the class order of `corn.yaml`
(`labels/`). `frames.csv` gives the source video, frame index and time of each frame.

For the test, each frame is taken again from the 8K video at the scale at which the rows appear as
large as in the training images. The scale is estimated on two fields and applied to the other
eight, in five rotations (`e12_external_test.py`). `paper_results/e12_external_test.json` gives
the out-of-fold mAP@50, far-class AP and mAP@50-95 of every training run.

The images and labels are released under CC BY 4.0.

## Licence

Builds on [Ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0) via
[YOLO-Pruning-RKNN](https://github.com/heyongxin233/YOLO-Pruning-RKNN); pruning uses
[Torch-Pruning](https://github.com/VainF/Torch-Pruning). Released under AGPL-3.0.

## Citation

Citation details will be added once the paper is published.
