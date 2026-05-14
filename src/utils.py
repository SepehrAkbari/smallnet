import torch
import numpy as np

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-8)

def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score and mIoU"""
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    iu = per_class_iu(hist)
    mean_iu = np.nanmean(iu)
    return acc, mean_iu