import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2


class DoubleConv(nn.Module):
    """Double convolution : (Conv2D -> ReLU) x 2"""
    
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=0)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=0)
        self.relu2 = nn.ReLU(inplace=True)

        # Initialisation des poids
        torch.nn.init.normal_(
            self.conv1.weight, mean=0, std=np.sqrt(2 / (3 * 3 * in_channels))
        )
        torch.nn.init.normal_(
            self.conv2.weight, mean=0, std=np.sqrt(2 / (3 * 3 * out_channels))
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        return x


class Down(nn.Module):
    """Module de descente : MaxPool -> DoubleConv"""
    
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.down = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)
        
    def forward(self, x):
        x = self.down(x)
        x = self.conv(x)
        return x


class Up(nn.Module):
    """Module de montée : ConvTranspose2D -> Concatenation -> DoubleConv"""
    
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(in_channels, out_channels)
        
    def forward(self, x, res):
        x = self.up(x)
        # Recadrer le résidu pour qu'il corresponde à x
        res = v2.CenterCrop(x.size()[-2:])(res)
        x = torch.cat([x, res], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    """
    Architecture U-Net pour la segmentation d'images.
    
    Args:
        n_channels (int): Nombre de canaux d'entrée
        n_classes (int): Nombre de classes pour la segmentation
    """
    
    def __init__(self, n_channels=1, n_classes=2):
        super(UNet, self).__init__()
        
        # Contracting path (encoder)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # Expansive path (decoder)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        # Output layer
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Contracting path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Expansive path
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
            
        # Output
        output = self.outc(x)
        return output


class UNetLoss(nn.Module):
    """Fonction de perte pour U-Net avec pondération des classes."""
    
    def __init__(self):
        super(UNetLoss, self).__init__()

    def forward(self, outputs, targets, weights=None):
        outputs = F.softmax(outputs, dim=1)
        outputs = outputs.view(*outputs.size()[:2], -1)
        targets = targets.view(*targets.size()[:1], -1)

        loss = F.cross_entropy(outputs, targets, reduction="none")
        
        if weights is not None:
            loss *= weights.view(*weights.size()[:1], -1)

        loss = loss.sum(axis=0).mean()
        
        return loss