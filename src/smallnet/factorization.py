'''
Layer factorization utilities for compression experiments.
'''

import torch
import torch.nn as nn
# import tltorch

from src.smallnet.modules import get_module, set_module


class MatrixLowRankConv2d(nn.Sequential):
    '''
    Matrix-SVD baseline for a Conv2d layer.

    The dense kernel is unfolded as [out_channels, in_channels * kh * kw],
    truncated to rank R, then implemented as a spatial Conv2d to R channels
    followed by a 1x1 Conv2d back to out_channels.
    '''

    @classmethod
    def from_conv(cls, conv, rank):
        if conv.groups != 1:
            raise ValueError("MatrixLowRankConv2d currently supports groups=1 only")

        weight = conv.weight.detach().cpu()
        out_channels, in_channels, kh, kw = weight.shape
        max_rank = min(out_channels, in_channels * kh * kw)
        rank = min(int(rank), max_rank)

        unfolded = weight.reshape(out_channels, -1)
        u, s, vh = torch.linalg.svd(unfolded, full_matrices=False)
        return cls.from_svd(conv, rank, u, s, vh)

    @classmethod
    def from_svd(cls, conv, rank, u, s, vh):
        '''Build the two-convolution baseline from a precomputed output-mode SVD.'''
        if conv.groups != 1:
            raise ValueError("MatrixLowRankConv2d currently supports groups=1 only")

        out_channels, in_channels, kh, kw = conv.weight.shape
        max_rank = min(out_channels, in_channels * kh * kw, len(s))
        rank = min(int(rank), max_rank)
        sqrt_s = torch.sqrt(s[:rank])

        first = nn.Conv2d(
            in_channels,
            rank,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            bias=False,
        )
        second = nn.Conv2d(rank, out_channels, kernel_size=1, bias=conv.bias is not None)

        first.weight.data.copy_((sqrt_s[:, None] * vh[:rank]).reshape(rank, in_channels, kh, kw))
        second.weight.data.copy_((u[:, :rank] * sqrt_s[None, :]).reshape(out_channels, rank, 1, 1))
        if conv.bias is not None:
            second.bias.data.copy_(conv.bias.detach().cpu())

        return cls(first, second)

    def composed_kernel(self):
        '''Return the exact dense kernel represented by the two convolutions.'''
        first, second = self
        out_channels, rank = second.weight.shape[:2]
        return torch.matmul(
            second.weight.reshape(out_channels, rank),
            first.weight.reshape(rank, -1),
        ).reshape(out_channels, first.in_channels, *first.kernel_size)


def factorized_conv_from_conv(conv, rank, factorization="cp", init="random", n_iter_max=0):
    if factorization == "cp":
        return tltorch.FactorizedConv.from_conv(
            conv,
            rank=int(rank),
            factorization="cp",
            decomposition_kwargs={"init": init, "n_iter_max": int(n_iter_max)},
        )
    if factorization in {"matrix", "svd", "matrix_svd"}:
        return MatrixLowRankConv2d.from_conv(conv, rank)
    raise ValueError(f"Unsupported factorization: {factorization}")


def replace_conv_layer(model, layer_name, rank, factorization="cp", init="random", n_iter_max=0):
    original = get_module(model, layer_name)
    if not isinstance(original, nn.Conv2d):
        raise TypeError(f"{layer_name} is not an nn.Conv2d: {type(original)}")
    replacement = factorized_conv_from_conv(
        original,
        rank=rank,
        factorization=factorization,
        init=init,
        n_iter_max=n_iter_max,
    )
    set_module(model, layer_name, replacement)
    return replacement


def load_dense_checkpoint(model, checkpoint_path, map_location="cpu"):
    state_dict = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(state_dict)
    return model


def build_factorized_model_from_dense(
    model,
    dense_checkpoint_path,
    layer_name,
    rank,
    factorization="cp",
    init="random",
    n_iter_max=0,
    save_path=None,
):
    '''
    Load a dense checkpoint, replace one Conv2d layer with a factorized layer,
    and optionally save the resulting state dict.
    '''

    load_dense_checkpoint(model, dense_checkpoint_path, map_location="cpu")
    replacement = replace_conv_layer(
        model,
        layer_name=layer_name,
        rank=rank,
        factorization=factorization,
        init=init,
        n_iter_max=n_iter_max,
    )
    if save_path:
        torch.save(model.state_dict(), save_path)
    return model, replacement


def infer_cp_rank_from_state_dict(state_dict, prefix):
    key = f"{prefix}.weight.weights"
    if key in state_dict:
        return int(state_dict[key].numel())
    return None
