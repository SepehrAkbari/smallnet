import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

class VGG16_FCN32s(nn.Module):
    def __init__(self, num_classes=32):
        super(VGG16_FCN32s, self).__init__()
        
        vgg = vgg16(weights=VGG16_Weights.DEFAULT)
        self.features = vgg.features
        
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, num_classes, kernel_size=1)
        )
        
        self.upsample = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = self.upsample(x)
        return x