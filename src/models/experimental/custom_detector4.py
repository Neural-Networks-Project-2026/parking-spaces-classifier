import torch
import torch.nn as nn
import torch.nn.functional as F

class AddCoords(nn.Module):
    """Dokleja kanały z informacją o współrzędnych (X, Y) do wejścia (CoordConv)"""
    def forward(self, x):
        b, c, h, w = x.size()
        
        # Tworzenie siatki współrzędnych znormalizowanych do przedziału [-1, 1]
        y_coords = torch.linspace(-1, 1, h, device=x.device).view(1, 1, h, 1).expand(b, 1, h, w)
        x_coords = torch.linspace(-1, 1, w, device=x.device).view(1, 1, 1, w).expand(b, 1, h, w)
        
        # Doklejamy 2 nowe kanały. Jeśli wejście miało 3 (RGB), wyjście będzie miało 5.
        return torch.cat([x, x_coords, y_coords], dim=1)

class SEResidualBlock(nn.Module):
    """Blok rezydualny z mechanizmem Squeeze-and-Excitation (Attention)"""
    def __init__(self, in_channels, out_channels, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        # Squeeze-and-Excitation 
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # "Squeeze" - globalne uśrednienie
            nn.Conv2d(out_channels, out_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // reduction, out_channels, 1, bias=False),
            nn.Sigmoid() # "Excitation" - wagi od 0 do 1 dla każdego kanału
        )

    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Aplikacja wagi SE
        se_weight = self.se(out)
        out = out * se_weight
        
        # Dodanie wejścia (Residual)
        out += identity
        return F.relu(out)

class AdvancedCenterNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        
        self.add_coords = AddCoords()
        
        # Enkoder 5 kanałowy
        self.inc = SEResidualBlock(5, 64)
        
        self.down1 = nn.Sequential(nn.MaxPool2d(2), SEResidualBlock(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), SEResidualBlock(128, 256))

        # dekoder
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up1 = SEResidualBlock(256 + 128, 128)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up2 = SEResidualBlock(128 + 64, 64)

       
        self.head_heatmap = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
        
        self.head_wh = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1)
        )

        # Inicjalizacja tła
        self.head_heatmap[-1].bias.data.fill_(-2.19)

    def forward(self, x):
        # Doklejenie współrzędnych do obrazka wejściowego!
        x = self.add_coords(x)
        
        # Koder
        x1 = self.inc(x)      
        x2 = self.down1(x1)   
        x3 = self.down2(x2)   

        # Dekoder
        u1 = self.up1(x3)
        u1 = torch.cat([x2, u1], dim=1) 
        u1 = self.conv_up1(u1) 
        
        u2 = self.up2(u1)
        u2 = torch.cat([x1, u2], dim=1)
        out_features = self.conv_up2(u2) 

        # Głowy
        heatmap = torch.clamp(torch.sigmoid(self.head_heatmap(out_features)), min=1e-4, max=1-1e-4)
        wh = F.relu(self.head_wh(out_features)) 

        return {"heatmap": heatmap, "wh": wh}