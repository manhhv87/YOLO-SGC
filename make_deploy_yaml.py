# -*- coding: utf-8 -*-
"""make_deploy_yaml.py — write the deployment YAML: drop ScaleMapHead and both ScaleMapDown layers.

Three layers are removed (indices 12, 13, 14), so every absolute index from 15 onwards
shifts down by three. Generated from a checked structure rather than edited by hand.
"""
import sys

HEAD_DEPLOY = """head:
- [9, 1, Conv, [256, 1, 1]]          # 10
- [-1, 1, AMRF, [256]]               # 11
- [4, 1, Conv, [128, 3, 2]]          # 12  (was 15)
- [6, 1, Conv, [128, 1, 1]]          # 13  (was 16)
- [11, 1, nn.Upsample, [None, 2, nearest]]   # 14  (was 17)
- [-1, 1, Conv, [128, 1, 1]]         # 15  (was 18)
- [[12, 13, 15], 1, {BLOCK}, [128]]  # 16  (was 19, scale-map input removed)
- [2, 1, Conv, [128, 3, 2]]          # 17  (was 20)
- [4, 1, Conv, [128, 1, 1]]          # 18  (was 21)
- [16, 1, nn.Upsample, [None, 2, nearest]]   # 19  (was 22)
- [-1, 1, Conv, [128, 1, 1]]         # 20  (was 23)
- [[17, 18, 20], 1, {BLOCK}, [128]]  # 21  (was 24)
- [[21, 16, 11], 1, Detect, [nc]]    # 22  (was 25)
"""

BACKBONE = """nc: 4
scales:
  n: [0.33, 0.25, 1024]
  s: [0.33, 0.5, 1024]
  m: [0.67, 0.75, 768]
  l: [1.0, 1.0, 512]
  x: [1.0, 1.25, 512]
backbone:
- [-1, 1, Conv, [64, 3, 2]]      # 0
- [-1, 1, Conv, [128, 3, 2]]     # 1
- [-1, 3, C2f, [128, true]]      # 2
- [-1, 1, Conv, [256, 3, 2]]     # 3
- [-1, 6, C2f, [256, true]]      # 4
- [-1, 1, Conv, [512, 3, 2]]     # 5
- [-1, 6, C2f, [512, true]]      # 6
- [-1, 1, Conv, [1024, 3, 2]]    # 7
- [-1, 3, C2f, [1024, true]]     # 8
- [-1, 1, SPPF, [1024, 5]]       # 9
"""

# old -> new layer index, for porting the state_dict
INDEX_MAP = {10: 10, 11: 11, 15: 12, 16: 13, 17: 14, 18: 15, 19: 16,
             20: 17, 21: 18, 22: 19, 23: 20, 24: 21, 25: 22}
DROPPED = {12, 13, 14}

if __name__ == "__main__":
    for tag, block in (("yprior_deploy", "SGCBlock_YPriorDeploy"), ("deploy", "SGCBlock_Deploy")):
        out = f"ultralytics/cfg/models/v8/yolov8s_sgc_p345_{tag}.yaml"
        open(out, "w").write(BACKBONE + HEAD_DEPLOY.replace("{BLOCK}", block))
        print("->", out)
