'''
Singular Value Decay of classifier layer.
'''

import torch
import matplotlib.pyplot as plt

from src.model import VGG16_FCN32s


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VGG16_FCN32s(num_classes=32, pretrained=False)
model.load_state_dict(torch.load("model/best_model.pth", map_location=device))

W = model.classifier[0].weight.detach()

print(f"Original Tensor Shape: {W.shape}")
W_unfolded = W.view(W.size(0), -1)
print(f"Unfolded Matrix Shape: {W_unfolded.shape}")

print("Computing SVD")
U, S, Vh = torch.linalg.svd(W_unfolded, full_matrices=False)
S_numpy = S.cpu().numpy()

plt.figure(figsize=(10, 6))
plt.plot(S_numpy[:512], linewidth=2.5, color='darkred') 

plt.axvline(x=64, color='black', linestyle='--', alpha=0.7, label='Rank-64 Bottleneck')
plt.axvline(x=128, color='grey', linestyle=':', alpha=0.7, label='Rank-128')
plt.axvline(x=256, color='grey', linestyle='-.', alpha=0.7, label='Rank-256')

plt.yscale('log')
plt.title("Singular Value Decay of classifier.0 (Mode-1 Unfolding)", fontsize=14, fontweight='bold')
plt.xlabel("Singular Value Index", fontsize=12)
plt.ylabel("Magnitude (Log Scale)", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.tight_layout()

plt.savefig("svd_decay.png", dpi=300)
plt.show()