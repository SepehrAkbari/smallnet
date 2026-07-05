'''
Generic segmentation evaluation utilities.
'''

import numpy as np
import torch

from src.utils import fast_hist, summarize_hist


def model_output_logits(output):
    if isinstance(output, dict) and "out" in output:
        return output["out"]
    return output


@torch.no_grad()
def evaluate_segmentation(model, loader, num_classes, device, ignore_index=None, class_names=None, max_batches=None):
    model = model.to(device)
    model.eval()
    hist = np.zeros((num_classes, num_classes), dtype=np.float64)
    if max_batches == 0:
        summary = summarize_hist(
            hist,
            class_names=class_names,
            exclude_indices=[ignore_index] if ignore_index is not None else None,
        )
        return hist, summary

    for batch_idx, (images, masks) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device)
        logits = model_output_logits(model(images))
        preds = logits.argmax(dim=1).cpu().numpy()
        gts = masks.numpy()
        hist += fast_hist(gts.flatten(), preds.flatten(), num_classes, ignore_index=ignore_index)

    summary = summarize_hist(
        hist,
        class_names=class_names,
        exclude_indices=[ignore_index] if ignore_index is not None else None,
    )
    return hist, summary
