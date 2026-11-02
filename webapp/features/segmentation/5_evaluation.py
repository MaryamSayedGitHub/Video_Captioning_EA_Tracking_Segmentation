"""STEP 5 - Evaluation of segmentation (needs ground-truth masks).

Mask IoU = |pred AND gt| / |pred OR gt|
Dice     = 2 |pred AND gt| / (|pred| + |gt|)
Mask mAP@50 / 50-95: COCO-style, from Ultralytics: model.val(data="your_dataset.yaml")
"""
import numpy as np


def mask_iou(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    union = np.logical_or(pred, gt).sum()
    return float(np.logical_and(pred, gt).sum() / union) if union else 1.0


def dice(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    s = pred.sum() + gt.sum()
    return float(2 * np.logical_and(pred, gt).sum() / s) if s else 1.0
