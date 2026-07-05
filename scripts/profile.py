'''
Manual parameter, MAC, and latency profiling for dense and CP-factorized models.
'''

import argparse
import csv
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import tltorch

from src.model import VGG16_FCN32s


def infer_cp_rank(state_dict):
    key = "classifier.0.weight.weights"
    if key in state_dict:
        return int(state_dict[key].numel())
    return None


def build_model(num_classes=32, rank=None):
    model = VGG16_FCN32s(num_classes=num_classes, pretrained=False)
    if rank is not None:
        model.classifier[0] = tltorch.FactorizedConv.from_conv(
            model.classifier[0],
            rank=rank,
            factorization="cp",
            decomposition_kwargs={"init": "random", "n_iter_max": 0},
        )
    return model


def load_model(checkpoint=None, num_classes=32, rank=None):
    if checkpoint is None:
        return build_model(num_classes=num_classes, rank=rank), rank

    state_dict = torch.load(checkpoint, map_location="cpu")
    inferred_rank = infer_cp_rank(state_dict)
    rank = inferred_rank if inferred_rank is not None else rank
    model = build_model(num_classes=num_classes, rank=rank)
    model.load_state_dict(state_dict)
    return model, rank


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def conv2d_macs(module, input_shape, output_shape):
    batch, _, out_h, out_w = output_shape
    cout = module.out_channels
    cin_per_group = module.in_channels // module.groups
    kh, kw = module.kernel_size
    return int(batch * out_h * out_w * cout * cin_per_group * kh * kw)


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


def manual_macs(model, input_size):
    records = []
    hooks = []

    def add_record(name, module, macs, output_shape):
        records.append(
            {
                "name": name,
                "type": module.__class__.__name__,
                "output_shape": list(output_shape),
                "params": sum(p.numel() for p in module.parameters()),
                "macs": macs,
            }
        )

    def make_hook(name):
        def hook(module, inputs, output):
            input_shape = tuple(inputs[0].shape)
            output_shape = tuple(output.shape)
            if module.__class__.__name__ == "FactorizedConv":
                macs = cp_conv2d_macs(module, input_shape, output_shape)
            elif isinstance(module, nn.Conv2d):
                macs = conv2d_macs(module, input_shape, output_shape)
            else:
                return
            add_record(name, module, macs, output_shape)

        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or module.__class__.__name__ == "FactorizedConv":
            hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
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


@torch.no_grad()
def measure_latency(model, input_size, device, warmup=20, iterations=100):
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

    return sum(timings) / len(timings)


def profile_model(label, checkpoint, rank, input_size, device, latency, warmup, iterations):
    model, actual_rank = load_model(checkpoint=checkpoint, rank=rank)
    model = model.to(device)
    macs, layer_records = manual_macs(model, input_size)
    latency_ms = measure_latency(model, input_size, device, warmup, iterations) if latency else None
    return {
        "label": label,
        "rank": "dense" if actual_rank is None else actual_rank,
        "checkpoint": checkpoint or "",
        "parameters": count_parameters(model),
        "macs": macs,
        "latency_ms": latency_ms,
        "layer_records": layer_records,
    }


def write_table(rows, out_path):
    if out_path is None:
        return
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["label", "rank", "checkpoint", "parameters", "macs", "latency_ms"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in writer.fieldnames})


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint to profile.")
    parser.add_argument("--ranks", type=int, nargs="*", default=[256, 128, 64])
    parser.add_argument("--height", type=int, default=352)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--latency", action="store_true")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--out", default="res/profile_manual.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(
        args.device
        if args.device
        else "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    input_size = (args.batch_size, 3, args.height, args.width)

    if args.checkpoint:
        rows = [
            profile_model(
                Path(args.checkpoint).stem,
                args.checkpoint,
                None,
                input_size,
                device,
                args.latency,
                args.warmup,
                args.iterations,
            )
        ]
    else:
        rows = [
            profile_model("Baseline", None, None, input_size, device, args.latency, args.warmup, args.iterations)
        ]
        for rank in args.ranks:
            rows.append(
                profile_model(
                    f"CP Rank-{rank}",
                    None,
                    rank,
                    input_size,
                    device,
                    args.latency,
                    args.warmup,
                    args.iterations,
                )
            )

    print(f"Device: {device.type}")
    print(f"{'Model':<18} {'Rank':<8} {'Params':>14} {'MACs (G)':>12} {'Latency (ms)':>14}")
    for row in rows:
        latency = "" if row["latency_ms"] is None else f"{row['latency_ms']:.2f}"
        print(
            f"{row['label']:<18} {str(row['rank']):<8} "
            f"{row['parameters']:>14,} {row['macs'] / 1e9:>12.2f} {latency:>14}"
        )
    write_table(rows, args.out)
    if args.out:
        print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
