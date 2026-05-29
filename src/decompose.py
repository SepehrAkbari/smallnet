import torch
import tltorch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.model import VGG16_FCN32s
from src.dataset import CamVidDataset
from src.utils import label_accuracy_score

def evaluate_decomposed_model(rank):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 32

    print(f"Loading baseline model on {device}...")
    model = VGG16_FCN32s(num_classes=num_classes)
    model.load_state_dict(torch.load("model/best_model.pth", map_location=device))
    
    # classifier[0] is Conv2d(512, 4096, 7, 7) layer
    original_layer = model.classifier[0]
    orig_params = sum(p.numel() for p in original_layer.parameters())
    
    decomposed_layer = tltorch.FactorizedConv.from_conv(
        original_layer, 
        rank=rank, 
        factorization='cp',
        decomposition_kwargs={'init': 'random'}
    )
    
    new_params = sum(p.numel() for p in decomposed_layer.parameters())
    print(f"Original Layer Params: {orig_params:,}")
    print(f"Decomposed Layer Params: {new_params:,} ({(new_params/orig_params)*100:.2f}%)")
    
    model.classifier[0] = decomposed_layer
    model = model.to(device)
    model.eval()

    t = transforms.Compose([
        transforms.Resize((352, 480)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dict_path = "data/CamVid/class_dict.csv"
    val_ds = CamVidDataset("data/CamVid/val", "data/CamVid/val_labels", dict_path, transform=t)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    print("\nEvaluating Zero-Shot Performance (No Retraining)...")
    preds, gts = [], []
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            outputs = model(images)
            pred = outputs.argmax(dim=1).cpu().numpy()
            preds.append(pred)
            gts.append(masks.numpy())

    acc, miou = label_accuracy_score(gts, preds, num_classes)
    print("="*40)
    print(f"Rank-{rank} Zero-Shot Val mIoU: {miou:.4f}")
    print(f"Rank-{rank} Zero-Shot Val Acc:  {acc:.4f}")
    print("="*40)

if __name__ == "__main__":
    # single aggressive rank
    evaluate_decomposed_model(rank=256)