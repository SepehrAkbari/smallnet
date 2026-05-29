import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import tltorch

from src.model import VGG16_FCN32s
from src.dataset import CamVidDataset
from src.train import Trainer

def finetune_decomposed_model(target_rank, epochs=10, log_file="logs/pareto_results.csv"):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    num_classes = 32
    
    # 1. Load the pristine baseline (pretrained=False to prevent cluster download hangs)
    model = VGG16_FCN32s(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load("model/best_model.pth", map_location=device))
    
    model = model.to(device)
    
    # 2. Decompose the massive layer
    print(f"\n--- Decomposing Layer to Rank {target_rank} ---")
    original_params = sum(p.numel() for p in model.classifier[0].parameters())
    
    decomposed_layer = tltorch.FactorizedConv.from_conv(
        model.classifier[0], 
        rank=target_rank, 
        factorization='cp',
        decomposition_kwargs={'init': 'random', 'n_iter_max': 10} 
    )
    
    model.classifier[0] = decomposed_layer
    new_params = sum(p.numel() for p in model.classifier[0].parameters())
    print(f"Layer Params: {original_params:,} -> {new_params:,} ({(new_params/original_params)*100:.2f}%)")
    
    # 3. Freeze the features
    # This forces the network to ONLY train the new factorized tensor components
    for param in model.features.parameters():
        param.requires_grad = False
        
    # 4. Data Setup
    t = transforms.Compose([
        transforms.Resize((352, 480)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dict_path = "data/CamVid/class_dict.csv"
    train_ds = CamVidDataset("data/CamVid/train", "data/CamVid/train_labels", dict_path, transform=t)
    val_ds = CamVidDataset("data/CamVid/val", "data/CamVid/val_labels", dict_path, transform=t)
    
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    # 5. Initialize the Trainer from Phase 1
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, device, num_classes)
    
    # 6. ZERO-SHOT EVALUATION
    print("\n--- Evaluating Zero-Shot Performance (No Retraining) ---")
    zs_acc, zs_miou = trainer.validate()
    print(f"Rank-{target_rank} Zero-Shot Val mIoU: {zs_miou:.4f}")
    print(f"Rank-{target_rank} Zero-Shot Val Acc:  {zs_acc:.4f}")

    # 7. Run the recovery!
    print(f"\n--- Starting Rank-{target_rank} Fine-Tuning ({epochs} Epochs) ---")
    save_path = f"model/finetuned_rank_{target_rank}.pth"
    trainer.fit(epochs, save_path=save_path)
    
    # 8. Log Results to CSV
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_exists = os.path.isfile(log_file)
    with open(log_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Rank", "Parameters", "Zero_Shot_Acc", "Zero_Shot_mIoU", "Fine_Tuned_mIoU"])
        # trainer.best_miou contains the highest mIoU achieved during the 10-epoch recovery
        writer.writerow([target_rank, new_params, zs_acc, zs_miou, trainer.best_miou])

if __name__ == "__main__":
    # The experimental ranks for the Pareto Frontier
    target_ranks = [2048, 1024, 512, 256, 128, 64]
    log_path = "logs/pareto_results.csv"
    
    for r in target_ranks:
        print(f"\n{'='*50}")
        print(f"COMMENCING EXPERIMENT FOR TENSOR RANK: {r}")
        print(f"{'='*50}")
        finetune_decomposed_model(target_rank=r, epochs=10, log_file=log_path)