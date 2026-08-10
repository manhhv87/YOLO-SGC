# YOLO-SGC — Scale-Guided Context Fusion Neck

Reference implementation for the paper

> **A Lightweight Scale-Guided Context Fusion Neck for Far-Field Crop-Row Detection on Edge Hardware**

The SGC neck replaces the FPN/PAN neck of YOLO detectors. Its three parts are in
`ultralytics/nn/modules/block.py`: an **AMRF** block that aggregates multi-receptive-field
context through dilated depthwise branches, a **ScaleMapHead** that derives a spatial prior
from early backbone features without depth supervision, and an **SGCBlock** that fuses
pyramid levels under spatially varying, feature-conditioned weights. Two configurations:
**SGC-P345** (three detection levels) and **SGC-P45** (drops the P3 branch, used for
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
**not** in this repository; they are available from the corresponding author on reasonable
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
python prune.py                              # structured pruning
python eval_guidance_line.py --split test    # guidance-line quality
python bench_all_fps.py                      # throughput
```

Training uses SGD (momentum 0.937, weight decay 5e-4), batch 32, cosine schedule with
initial LR 0.005, 640×640 input, at most 200 epochs with patience 30. Accuracy comparisons
are means over five seeds (0–4). `paper_results/*.json` holds the per-checkpoint accuracy
metrics behind the tables.

## Licence

Builds on [Ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0) via
[YOLO-Pruning-RKNN](https://github.com/heyongxin233/YOLO-Pruning-RKNN); pruning uses
[Torch-Pruning](https://github.com/VainF/Torch-Pruning). Released under **AGPL-3.0**.

## Citation

Citation details will be added once the paper is published.
