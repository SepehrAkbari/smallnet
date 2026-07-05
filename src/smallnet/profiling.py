'''
Manual parameter, MAC, and latency accounting.
'''

import time

import numpy as np
import torch
import torch.nn as nn

from src.smallnet.modules import count_parameters


def conv2d_macs(module, input_shape, output_shape):
    batch, _, out_h, out_w = output_shape
    cout = module.out_channels
    cin_per_group = module.in_channels // module.groups
    kh, kw = module.kernel_size
    return int(batch * out_h * out_w * cout * cin_per_group * kh * kw)


def get_cp_rank(module):
    rank = getattr(module, "rank", None)
    if isinstance(rank, int):
        return rank
    weight = getattr(module, "weight", None)
    rank = getattr(weight, "rank", None)
    if isinstance(rank, int):
        return rank
    state = module.state_dict()
    if "weight.weights" in state:
        return int(state["weight.weights"].numel())
    raise ValueError(f"Could not infer CP rank for {module}")


def is_factorized_conv(module):
    return module.__class__.__name__ == "FactorizedConv"


def cp_conv2d_macs(module, input_shape, output_shape):
    batch, cin, _, _ = input_shape
    _, cout, out_h, out_w = output_shape
    kh, kw = module.kernel_size
    rank = get_cp_rank(module)
    pointwise_in = batch * out_h * out_w * cin * rank
    vertical = batch * out_h * out_w * rank * kh
    horizontal = batch * out_h * out_w * rank * kw
    pointwise_out = batch * out_h * out_w * rank * cout
    cp_weights = batch * out_h * out_w * rank
    return int(pointwise_in + vertical + horizontal + pointwise_out + cp_weights)


def manual_macs(model, input_size, device=None):
    records = []
    hooks = []

    def make_hook(name):
        def hook(module, inputs, output):
            input_shape = tuple(inputs[0].shape)
            output_shape = tuple(output.shape)
            if is_factorized_conv(module):
                macs = cp_conv2d_macs(module, input_shape, output_shape)
            elif isinstance(module, nn.Conv2d):
                macs = conv2d_macs(module, input_shape, output_shape)
            else:
                return
            records.append(
                {
                    "name": name,
                    "type": module.__class__.__name__,
                    "output_shape": list(output_shape),
                    "params": count_parameters(module),
                    "macs": macs,
                }
            )

        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or is_factorized_conv(module):
            hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    if device is None:
        device = next(model.parameters()).device
    dummy = torch.randn(*input_size, device=device)
    with torch.no_grad():
        model(dummy)

    for hook in hooks:
        hook.remove()

    return sum(row["macs"] for row in records), records


def sync_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def summarize_latency_timings(timings, warmup, iterations, device_name=""):
    values = np.asarray(timings, dtype=np.float64)
    if values.size == 0:
        return {
            "latency_mean_ms": None,
            "latency_std_ms": None,
            "latency_median_ms": None,
            "latency_min_ms": None,
            "latency_max_ms": None,
            "latency_warmup_iterations": int(warmup),
            "latency_iterations": int(iterations),
            "device_name": device_name,
        }
    return {
        "latency_mean_ms": float(values.mean()),
        "latency_std_ms": float(values.std(ddof=0)),
        "latency_median_ms": float(np.median(values)),
        "latency_min_ms": float(values.min()),
        "latency_max_ms": float(values.max()),
        "latency_warmup_iterations": int(warmup),
        "latency_iterations": int(iterations),
        "device_name": device_name,
    }


@torch.no_grad()
def latency_stats(model, input_size, device, warmup=20, iterations=100, device_name=""):
    model = model.to(device)
    model.eval()
    dummy = torch.randn(*input_size, device=device)
    for _ in range(warmup):
        model(dummy)
    sync_device(device)

    timings = []
    for _ in range(iterations):
        sync_device(device)
        start = time.perf_counter()
        model(dummy)
        sync_device(device)
        timings.append((time.perf_counter() - start) * 1000)
    return summarize_latency_timings(timings, warmup=warmup, iterations=iterations, device_name=device_name)


@torch.no_grad()
def latency_ms(model, input_size, device, warmup=20, iterations=100):
    stats = latency_stats(model, input_size, device, warmup=warmup, iterations=iterations)
    return stats["latency_mean_ms"]
