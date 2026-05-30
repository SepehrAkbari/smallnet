'''
Evaluating decomposition efficiency.
'''

import torch
import time
from thop import profile
import tltorch

from src.model import VGG16_FCN32s


def measure_efficiency(model, device):
    model = model.to(device)
    model.eval()
    
    dummy_input = torch.randn(1, 3, 352, 480).to(device)
    
    macs, params = profile(model, inputs=(dummy_input,), verbose=False)
    
    def sync_hardware():
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
            
    with torch.no_grad(): # warmup
        for _ in range(20):
            _ = model(dummy_input)
            
    iterations = 100
    timings = []
    
    with torch.no_grad():
        for _ in range(iterations):
            sync_hardware()
            start_time = time.perf_counter()
            
            _ = model(dummy_input)
            
            sync_hardware()
            end_time = time.perf_counter()
            
            timings.append((end_time - start_time) * 1000)
            
    avg_latency = sum(timings) / iterations
    return macs, params, avg_latency

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device.type.upper()}\n")
    
    print(f"{'Model / Rank':<15} | {'Parameters':<12} | {'MACs (G)':<10} | {'Latency (ms)':<15}")
    print("-" * 60)
    
    baseline = VGG16_FCN32s(num_classes=32, pretrained=False)
    base_macs, base_params, base_lat = measure_efficiency(baseline, device)
    print(f"{'Baseline':<15} | {base_params / 1e6:>5.2f} M    | {base_macs / 1e9:>5.2f} G  | {base_lat:>5.2f} ms")
    
    target_ranks = [256, 128, 64]
    
    for rank in target_ranks:
        model = VGG16_FCN32s(num_classes=32, pretrained=False)
        model = model.to(device)
        
        model.classifier[0] = tltorch.FactorizedConv.from_conv(
            model.classifier[0], 
            rank=rank, 
            factorization='cp',
            decomposition_kwargs={'init': 'random', 'n_iter_max': 0}
        )
        
        macs, params, lat = measure_efficiency(model, device)
        print(f"{f'CP Rank-{rank}':<15} | {params / 1e6:>5.2f} M    | {macs / 1e9:>5.2f} G  | {lat:>5.2f} ms")
        
        del model