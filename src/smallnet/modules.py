'''
Utilities for dotted module access, replacement, and layer summaries.
'''

from dataclasses import dataclass

import torch.nn as nn


@dataclass
class LayerSummary:
    name: str
    module_type: str
    params: int
    weight_shape: list[int]


def _split_path(path):
    return [part for part in path.split(".") if part]


def get_module(model, path):
    current = model
    for part in _split_path(path):
        if part.isdigit() and isinstance(current, (nn.Sequential, nn.ModuleList)):
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


def set_module(model, path, replacement):
    parts = _split_path(path)
    if not parts:
        raise ValueError("Cannot replace the root module")
    parent_path = ".".join(parts[:-1])
    parent = get_module(model, parent_path) if parent_path else model
    leaf = parts[-1]
    if leaf.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(leaf)] = replacement
    else:
        setattr(parent, leaf, replacement)
    return model


def count_parameters(module, trainable_only=False):
    params = module.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def iter_conv2d(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            yield name, module


def summarize_conv_layers(model):
    rows = []
    for name, module in iter_conv2d(model):
        rows.append(
            LayerSummary(
                name=name,
                module_type=module.__class__.__name__,
                params=count_parameters(module),
                weight_shape=list(module.weight.shape),
            )
        )
    return rows


def select_top_conv_layers(model, limit=3, min_params=1):
    rows = [row for row in summarize_conv_layers(model) if row.params >= min_params]
    rows.sort(key=lambda row: row.params, reverse=True)
    return rows[:limit]
