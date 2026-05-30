'''
Identifying parameters in model.
'''

from src.model import VGG16_FCN32s


model = VGG16_FCN32s(num_classes=32)

print("\n" + "-"*85)
print(f"{'Layer Name':<30} | {'Tensor Shape (Out, In, H, W)':<30} | {'Parameters':<15}")
print("-"*85)

total_4d_params = 0
heavy_layers = []

for name, param in model.named_parameters():
    if 'weight' in name and len(param.shape) == 4:
        params = param.numel()
        total_4d_params += params
        
        shape_str = str(list(param.shape))
        print(f"{name:<30} | {shape_str:<30} | {params:<15,}")
        
        if params > 10_000_000:
            heavy_layers.append((name, params))

print("-"*85)
print(f"Total 4D Tensor Parameters: {total_4d_params:,}")
print("\nTarget(s):")
for name, params in heavy_layers:
    print(f"{name}: {params:,} parameters ({(params/total_4d_params)*100:.1f}% of network)")