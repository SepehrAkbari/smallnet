'''
Visualization for qualitative comparison.
'''

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/smallnet-cache")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/smallnet-matplotlib")
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
import tltorch

from src.model import VGG16_FCN32s


def decode_segmap(pred_mask, class_colors):
    r = np.zeros_like(pred_mask).astype(np.uint8)
    g = np.zeros_like(pred_mask).astype(np.uint8)
    b = np.zeros_like(pred_mask).astype(np.uint8)
    
    for l in range(len(class_colors)):
        idx = pred_mask == l
        r[idx] = class_colors[l, 0]
        g[idx] = class_colors[l, 1]
        b[idx] = class_colors[l, 2]
        
    return np.stack([r, g, b], axis=2)

def load_model(rank, device, weights_path):
    model = VGG16_FCN32s(num_classes=32, pretrained=False)
    
    if rank is not None:
        model.classifier[0] = tltorch.FactorizedConv.from_conv(
            model.classifier[0], 
            rank=rank, 
            factorization='cp',
            decomposition_kwargs={'init': 'random', 'n_iter_max': 0}
        )
    
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_dict = pd.read_csv("data/CamVid/class_dict.csv")
    colors = class_dict[['r', 'g', 'b']].values

    t = transforms.Compose([
        transforms.Resize((352, 480)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_name = "Seq05VD_f03870.png" 
    img_path = os.path.join("data/CamVid/val", img_name)
    
    raw_img = Image.open(img_path).convert("RGB")
    raw_img_resized = raw_img.resize((480, 352))
    input_tensor = t(raw_img).unsqueeze(0).to(device)

    print("Loading Baseline...")
    base_model = load_model(None, device, "model/best_model.pth")
    
    print("Loading Rank-256...")
    r256_model = load_model(256, device, "model/finetuned_rank_256.pth")
    
    print("Loading Rank-64...")
    r64_model = load_model(64, device, "model/finetuned_rank_64.pth")

    print("Running Inference...")
    with torch.no_grad():
        base_pred = torch.argmax(base_model(input_tensor), dim=1).squeeze(0).cpu().numpy()
        r256_pred = torch.argmax(r256_model(input_tensor), dim=1).squeeze(0).cpu().numpy()
        r64_pred  = torch.argmax(r64_model(input_tensor), dim=1).squeeze(0).cpu().numpy()

    base_rgb = decode_segmap(base_pred, colors)
    r256_rgb = decode_segmap(r256_pred, colors)
    r64_rgb  = decode_segmap(r64_pred, colors)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(raw_img_resized)
    axes[0].set_title("Original Image", fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(base_rgb)
    axes[1].set_title("Baseline (Dense)", fontsize=14)
    axes[1].axis('off')
    
    axes[2].imshow(r256_rgb)
    axes[2].set_title("Rank-256 (1.18M Params)", fontsize=14)
    axes[2].axis('off')
    
    axes[3].imshow(r64_rgb)
    axes[3].set_title("Rank-64 (0.29M Params)", fontsize=14)
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig("res/qualitative_results.png", dpi=300, bbox_inches='tight')
    plt.close()
