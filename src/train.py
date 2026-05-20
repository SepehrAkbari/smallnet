import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os

from src.model import VGG16_FCN32s
from src.dataset import CamVidDataset
from src.utils import label_accuracy_score

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, num_classes):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.num_classes = num_classes
        self.best_miou = 0.0

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for images, masks in pbar:
            images, masks = images.to(self.device), masks.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            loss = self.criterion(outputs, masks)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            
        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        preds, gts = [], []
        
        for images, masks in self.val_loader:
            images = images.to(self.device)
            outputs = self.model(images)
            
            pred = outputs.argmax(dim=1).cpu().numpy()
            preds.append(pred)
            gts.append(masks.numpy())
            
        acc, miou = label_accuracy_score(gts, preds, self.num_classes)
        return acc, miou

    def fit(self, epochs, save_path="model/best_model.pth"):
        os.makedirs("model", exist_ok=True)
        
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(epoch)
            acc, miou = self.validate()
            
            print(f"-> Train Loss: {train_loss:.4f} | Val Acc: {acc:.4f} | Val mIoU: {miou:.4f}")
            
            if miou > self.best_miou:
                self.best_miou = miou
                torch.save(self.model.state_dict(), save_path)
                print(f"*** New Best mIoU: {miou:.4f}. Model saved. ***")

def run_experiment():
    # Hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    num_classes = 32
    batch_size = 4
    lr = 1e-4
    epochs = 200

    t = transforms.Compose([
        transforms.Resize((352, 480)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dict_path = "data/CamVid/class_dict.csv"
    train_ds = CamVidDataset("data/CamVid/train", "data/CamVid/train_labels", dict_path, transform=t)
    val_ds = CamVidDataset("data/CamVid/val", "data/CamVid/val_labels", dict_path, transform=t)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = VGG16_FCN32s(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, device, num_classes)
    trainer.fit(epochs)

if __name__ == "__main__":
    run_experiment()