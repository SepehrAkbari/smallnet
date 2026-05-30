import torch
import time
from thop import profile
import tltorch

from src.model import VGG16_FCN32s

def measure_efficiency(model, device):
    model = model.to(device)
    model.eval()
    
    # Standard CamVid resolution dummy input (Batch Size 1)
    dummy_input = torch.randn(1, 3, 352, 480).to(device)
    
    # 1. Profile MACs (Multiply-Accumulates) and Parameters
    macs, params = profile(model, inputs=(dummy_input,), verbose=False)
    
    # Helper function to ensure the hardware finishes computing before the timer stops
    def sync_hardware():
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
            
    # 2. Warm-up the hardware
    with torch.no_grad():
        for _ in range(20):
            _ = model(dummy_input)
            
    # 3. Measure Latency using device-agnostic timing
    iterations = 100
    timings = []
    
    with torch.no_grad():
        for _ in range(iterations):
            sync_hardware()
            start_time = time.perf_counter()
            
            _ = model(dummy_input)
            
            sync_hardware()
            end_time = time.perf_counter()
            
            # Convert to milliseconds
            timings.append((end_time - start_time) * 1000)
            
    avg_latency = sum(timings) / iterations
    return macs, params, avg_latency

def run_benchmarks():
    # Automatically select the best available hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running benchmarks on: {device.type.upper()}\n")
    
    print(f"{'Model / Rank':<15} | {'Parameters':<12} | {'MACs (G)':<10} | {'Latency (ms)':<15}")
    print("-" * 60)
    
    # 1. Baseline
    baseline = VGG16_FCN32s(num_classes=32, pretrained=False)
    base_macs, base_params, base_lat = measure_efficiency(baseline, device)
    print(f"{'Baseline':<15} | {base_params / 1e6:>5.2f} M    | {base_macs / 1e9:>5.2f} G  | {base_lat:>5.2f} ms")
    
    # 2. Decomposed Ranks
    target_ranks = [256, 128, 64]
    
    for rank in target_ranks:
        model = VGG16_FCN32s(num_classes=32, pretrained=False)
        model.classifier[0] = tltorch.FactorizedConv.from_conv(
            model.classifier[0], 
            rank=rank, 
            factorization='cp',
            decomposition_kwargs={'init': 'random', 'n_iter_max': 10}
        )
        
        macs, params, lat = measure_efficiency(model, device)
        print(f"{f'CP Rank-{rank}':<15} | {params / 1e6:>5.2f} M    | {macs / 1e9:>5.2f} G  | {lat:>5.2f} ms")

if __name__ == "__main__":
    run_benchmarks()