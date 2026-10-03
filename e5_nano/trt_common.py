# -*- coding: utf-8 -*-
"""trt_common.py - pre- and post-processing shared by the Jetson Nano (TensorRT) and PC scripts.

Runs on Python 3.6 + numpy 1.13 (the system Python of JetPack 4.6) as well as on a PC.
The Nano and the PC call the same functions, so an accuracy difference between the FP16 engine on
the Nano and the FP32 ONNX model on the PC comes only from the arithmetic of the engine.

Post-processing reproduces ultralytics/utils/ops.py::non_max_suppression with the parameters of
DetectionValidator (version 8.3.160 in this repository): conf 0.001, IoU 0.7, multi_label=True,
max_nms 30000, max_wh 7680 (class offset), max_det 300.
"""
import numpy as np

IMGSZ = 640
CONF = 0.001
IOU = 0.7
MAX_DET = 300
MAX_NMS = 30000
MAX_WH = 7680


def letterbox(img, new=IMGSZ, color=114):
    """Ultralytics LetterBox(new_shape=(new,new), auto=False, scaleup=False).

    The test images are 640x640, so the image is returned unchanged; the general case is kept for safety.
    Returns the padded image, the ratio r and the padding (left, top).
    """
    import cv2
    h0, w0 = img.shape[:2]
    r = min(float(new) / h0, float(new) / w0, 1.0)          # scaleup=False
    w1, h1 = int(round(w0 * r)), int(round(h0 * r))
    if (w1, h1) != (w0, h0):
        img = cv2.resize(img, (w1, h1), interpolation=cv2.INTER_LINEAR)
    dw, dh = (new - w1) / 2.0, (new - h1) / 2.0
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    if top or bottom or left or right:
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(color, color, color))
    return img, r, (left, top)


def preprocess(bgr):
    """BGR uint8 HxWx3 -> float32 1x3x640x640 in [0,1], RGB, plus what is needed to map boxes back to the image."""
    img, r, pad = letterbox(bgr)
    x = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.ascontiguousarray(x[None]), r, pad


def _iou_one(box, boxes):
    """IoU of one box with several boxes (x1,y1,x2,y2), as torchvision.ops.box_iou."""
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (a + b - inter)


def nms(boxes, scores, iou_thres, max_keep):
    """Greedy NMS in order of decreasing score; stops once max_keep boxes are kept.

    Ultralytics takes i[:max_det] from torchvision.ops.nms (also greedy by decreasing score),
    so stopping after the first max_det boxes gives identical results.
    """
    order = np.argsort(-scores, kind="mergesort")
    keep = []
    while order.size and len(keep) < max_keep:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = _iou_one(boxes[i], boxes[rest])
        order = rest[~(iou > iou_thres)]           # as torchvision: suppress only if IoU > threshold (NaN kept)
    return np.array(keep, dtype=np.int64)


def postprocess(out, shape0, r=1.0, pad=(0, 0), conf=CONF, iou=IOU, max_det=MAX_DET):
    """out: (1, 4+nc, N) float32, xywh boxes in input-image pixels; shape0 = (h0, w0) of the original image.

    Returns (k,6): x1,y1,x2,y2,conf,cls in original-image coordinates, clipped to the image as clip_boxes.
    """
    p = out[0].T                                    # (N, 4+nc)
    nc = p.shape[1] - 4
    cls = p[:, 4:]
    ai, ci = np.nonzero(cls > conf)                 # multi_label: every (anchor, class) pair above the threshold
    if ai.size == 0:
        return np.zeros((0, 6), np.float32)
    xywh = p[ai, :4]
    xyxy = np.empty_like(xywh)
    xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
    sc = cls[ai, ci]
    if sc.size > MAX_NMS:
        top = np.argsort(-sc, kind="mergesort")[:MAX_NMS]
        xyxy, sc, ci = xyxy[top], sc[top], ci[top]
    off = xyxy + (ci[:, None] * MAX_WH).astype(xyxy.dtype)
    k = nms(off, sc, iou, max_det)
    det = np.concatenate([xyxy[k], sc[k, None], ci[k, None].astype(np.float32)], 1).astype(np.float32)
    # back to original-image coordinates, then clip to the image (Ultralytics scale_boxes + clip_boxes)
    det[:, [0, 2]] -= pad[0]
    det[:, [1, 3]] -= pad[1]
    det[:, :4] /= r
    det[:, [0, 2]] = det[:, [0, 2]].clip(0, shape0[1])
    det[:, [1, 3]] = det[:, [1, 3]].clip(0, shape0[0])
    return det
