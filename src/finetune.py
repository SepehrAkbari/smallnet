import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import tltorch

from src.model import VGG16_FCN32s
from src.data_loader import CamVidDataset
from src.train import Trainer

def finetune_decomposed_model(target_rank, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 32
    
    # 1. Load the pristine baseline
    model = VGG16_FCN32s(num_classes=num_classes)
    model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
    
    # 2. Decompose the massive layer
    print(f"\n--- Decomposing Layer to Rank {target_rank} ---")
    decomposed_layer = tltorch.FactorizedConv.from_conv(
        model.classifier[0], 
        rank=target_rank, 
        factorization='cp',
        decomposition_kwargs={'init': 'random'}
    )
    model.classifier[0] = decomposed_layer
    
    # 3. Freeze the features (Optional, but highly recommended for 1-month timelines)
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
    # Notice we only pass parameters that require gradients to the optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, device, num_classes)
    
    # 6. Run the recovery!
    print(f"\n--- Starting Rank-{target_rank} Fine-Tuning ({epochs} Epochs) ---")
    save_path = f"models/finetuned_rank_{target_rank}.pth"
    trainer.fit(epochs, save_path=save_path)

if __name__ == "__main__":
    # You will loop this over your chosen ranks
    finetune_decomposed_model(target_rank=256, epochs=10)