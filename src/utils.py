'''
Utils for computing accuracy and mIoU.
'''

import numpy as np


def fast_hist(a, b, n, ignore_index=None):
    a = np.asarray(a)
    b = np.asarray(b)
    k = (a >= 0) & (a < n)
    if ignore_index is not None:
        k &= a != ignore_index
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)


def per_class_iu(hist, absent_score=0.0):
    denom = hist.sum(1) + hist.sum(0) - np.diag(hist)
    iu = np.divide(
        np.diag(hist),
        denom,
        out=np.full_like(denom, absent_score, dtype=np.float64),
        where=denom > 0,
    )
    return iu


def pixel_accuracy(hist):
    total = hist.sum()
    if total == 0:
        return 0.0
    return np.diag(hist).sum() / total


def frequency_weighted_iou(hist):
    total = hist.sum()
    if total == 0:
        return 0.0
    freq = hist.sum(1) / total
    iu = per_class_iu(hist, absent_score=0.0)
    return (freq[freq > 0] * iu[freq > 0]).sum()


def mean_iou(hist, absent_score=0.0):
    iu = per_class_iu(hist, absent_score=absent_score)
    if np.isnan(iu).all():
        return 0.0
    return np.nanmean(iu)


def summarize_hist(hist, class_names=None, exclude_indices=None):
    iu = per_class_iu(hist, absent_score=np.nan)
    iu_all = per_class_iu(hist, absent_score=0.0)
    support = hist.sum(1)
    names = class_names or [str(i) for i in range(hist.shape[0])]
    exclude_indices = set(exclude_indices or [])
    include = np.array([idx not in exclude_indices for idx in range(hist.shape[0])])

    total = hist.sum()
    freq = support / total if total > 0 else np.zeros_like(support)
    present_iu = iu[include]
    if np.isnan(present_iu).all():
        mean_present = 0.0
    else:
        mean_present = np.nanmean(present_iu)

    per_class = []
    for idx, name in enumerate(names):
        per_class.append(
            {
                "class_id": idx,
                "class_name": name,
                "support": int(support[idx]),
                "iou": None if np.isnan(iu[idx]) else float(iu[idx]),
                "excluded": idx in exclude_indices,
            }
        )

    return {
        "pixel_accuracy": float(pixel_accuracy(hist)),
        "mean_iou_all_classes": float(np.nanmean(iu_all[include])) if include.any() else 0.0,
        "mean_iou_present_classes": float(mean_present),
        "frequency_weighted_iou": float((freq[include & (freq > 0)] * iu_all[include & (freq > 0)]).sum()),
        "per_class": per_class,
    }


def label_accuracy_score(label_trues, label_preds, n_class):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = pixel_accuracy(hist)
    mean_iu = mean_iou(hist, absent_score=0.0)
    return acc, mean_iu
