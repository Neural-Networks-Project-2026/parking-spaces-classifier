import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
    
class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            # Konwolucja ze stride=2 pomniejsza wymiary o połowę zachowując relacje przestrzenne
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)

class SimpleUNetCenterNet(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        
        # Enkoder
        self.inc = DoubleConv(3, 64)
        # Zastąpione MaxPool na warstwy uczące się (DownsampleBlock)
        self.down1 = DownsampleBlock(64, 128)
        self.down2 = DownsampleBlock(128, 256)

        # Dekoder
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up1 = DoubleConv(256 + 128, 128)        
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up2 = DoubleConv(128 + 64, 64)

        
        # Głowa Heatmapy: Przewiduje szansę wystąpienia środka obiektu.
        # Im bliżej do środka obiektu - miejsca parkingowego, tym bliżej 1 
        # czyli 100% pewności 
        # Im dalej, tym bliżej 0 - czyli tło 
        self.head_heatmap = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
        
        # Głowa Rozmiaru (WH): Przewiduje absolutną szerokość i wysokość miejsca parkingowego.
        self.head_wh = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1)
        )

        # Głowa offsetu, potrzebna do tego aby gdy pomniejszamy obrazek (enkoder)
        # a potem go powiększamy (dekoder), to ona ma za zadanie sprawić by predicty
        # się nie rozjechały na boki
        self.head_offset = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1)
        )

        # Inicjalizacja wag dla mapy cieplnej 
        # Chcemy, aby sieć na początku przewidywała bardzo małe prawdopodobieństwa tła,
        # w przeciwnym razie strata wybuchnie, bo większość obrazu to tło
        self.head_heatmap[-1].bias.data.fill_(-2.19) 

    def forward(self, x):
        x1 = self.inc(x) 
        x2 = self.down1(x1)
        x3 = self.down2(x2)   

        u1 = self.up1(x3)
        u1 = torch.cat([x2, u1], dim=1) 
        u1 = self.conv_up1(u1) 
        
        u2 = self.up2(u1)
        u2 = torch.cat([x1, u2], dim=1) 
        out_features = self.conv_up2(u2) 
        
        heatmap = torch.clamp(torch.sigmoid(self.head_heatmap(out_features)), min=1e-4, max=1-1e-4)
        wh = F.relu(self.head_wh(out_features)) 
        
        offset = self.head_offset(out_features)

        return {"heatmap": heatmap, "wh": wh, "offset": offset}

def create_centernet_targets(boxes, labels, output_h, output_w, device):
    num_classes = 2
    hm = torch.zeros((num_classes, output_h, output_w), dtype=torch.float32, device=device)
    wh = torch.zeros((2, output_h, output_w), dtype=torch.float32, device=device)
    
    # NOWOŚĆ: tensor pod przesunięcia sub-pikselowe
    offset = torch.zeros((2, output_h, output_w), dtype=torch.float32, device=device)
    reg_mask = torch.zeros((output_h, output_w), dtype=torch.float32, device=device) 

    if boxes.numel() == 0:
        return hm, wh, offset, reg_mask

    for box, label in zip(boxes, labels):
        if label.item() == 0:
            continue
            
        class_idx = label.item() - 1 

        x1, y1, x2, y2 = box
        h, w = y2 - y1, x2 - x1
        
        # Prawdziwy, zmiennoprzecinkowy środek
        ctx, cty = x1 + w / 2, y1 + h / 2
        
        # Skwantyzowany, całkowity środek dla siatki
        ctx_int, cty_int = int(ctx.item()), int(cty.item())
        
        if ctx_int < 0 or ctx_int >= output_w or cty_int < 0 or cty_int >= output_h:
            continue

        radius = max(2, int(math.sqrt(h * w) / 6)) 
        draw_umich_gaussian(hm[class_idx], (ctx_int, cty_int), radius)

        wh[0, cty_int, ctx_int] = w
        wh[1, cty_int, ctx_int] = h
        
        # Obliczenie błędu zaokrąglenia i zapisanie go do mapy targetów
        offset[0, cty_int, ctx_int] = ctx.item() - ctx_int
        offset[1, cty_int, ctx_int] = cty.item() - cty_int
        
        reg_mask[cty_int, ctx_int] = 1

    return hm, wh, offset, reg_mask

def draw_umich_gaussian(heatmap, center, radius, k=1):
    """Pomocnicza funkcja z oryginalnej implementacji CenterNetu do rysowania rozmycia Gaussa"""
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    
    gaussian = gaussian.to(heatmap.device)
    
    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]
    
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def gaussian2D(shape, sigma=1):
    # generuje dzwon gaussowski dla 2d
    m, n = [(ss - 1.) / 2. for ss in shape]
    
    
    y = torch.arange(-m, m + 1, dtype=torch.float32).view(-1, 1)
    x = torch.arange(-n, n + 1, dtype=torch.float32).view(1, -1)
    
    h = torch.exp(-(x * x + y * y) / (2 * sigma * sigma))
    
    # wartości bliskie zera są zamieniane na 0
    h[h < torch.finfo(h.dtype).eps * h.max()] = 0
    return h


def focal_loss(pred, gt):
    """Zoptymalizowana Focal Loss dla oszacowywania map cieplnych z CenterNet"""
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)
    loss = 0

    # pixel po pixelu, karzemy za złe wskazania mapy cieplnej z modelu względem ground truth
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    # To samo, ale dla tła, a nie dla boxów. Tutaj karzemy mniej, bo mnożymy przez neg_weights
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss

def reg_l1_loss(pred, gt, mask):
    """Strata L1 dla rozmiarów (WH), liczona TYLKO tam, gdzie są obiekty (mask=1)"""
    mask = mask.unsqueeze(1).expand_as(pred) 
    loss = F.l1_loss(pred * mask, gt * mask, reduction='sum')
    loss = loss / (mask.sum() + 1e-4) 
    return loss

def decode_predictions(pred_hm, pred_wh, pred_offset, threshold=0.3):
    """
    Zamienia mapy cieplne na bounding boxy, uwzględniając poprawkę z głowy offset.
    """
    B, C, H, W = pred_hm.shape
    results = []

    pool = torch.nn.functional.max_pool2d(pred_hm, kernel_size=3, stride=1, padding=1)
    keep = (pool == pred_hm).float()
    pred_hm = pred_hm * keep

    for b in range(B):
        hm = pred_hm[b]
        wh = pred_wh[b]
        offset = pred_offset[b] # [2, H, W]

        boxes = []
        scores = []
        labels = []

        for class_idx in range(C):
            class_hm = hm[class_idx]
            ys, xs = torch.where(class_hm > threshold)

            for y, x in zip(ys, xs):
                score = class_hm[y, x].item()
                w = wh[0, y, x].item()
                h = wh[1, y, x].item()

                dx = offset[0, y, x].item()
                dy = offset[1, y, x].item()

                # Prawdziwy, skorygowany środek
                center_x = x.item() + dx
                center_y = y.item() + dy

                # Wyliczanie krawędzi za pomocą skorygowanego środka
                x_min = center_x - w / 2
                y_min = center_y - h / 2
                x_max = center_x + w / 2
                y_max = center_y + h / 2

                boxes.append([x_min, y_min, x_max, y_max])
                scores.append(score)
                labels.append(class_idx + 1)

        if len(boxes) > 0:
            results.append({
                "boxes": torch.tensor(boxes, dtype=torch.float32, device=pred_hm.device),
                "scores": torch.tensor(scores, dtype=torch.float32, device=pred_hm.device),
                "labels": torch.tensor(labels, dtype=torch.int64, device=pred_hm.device)
            })
        else:
            results.append({
                "boxes": torch.empty((0, 4), dtype=torch.float32, device=pred_hm.device),
                "scores": torch.empty((0,), dtype=torch.float32, device=pred_hm.device),
                "labels": torch.empty((0,), dtype=torch.int64, device=pred_hm.device)
            })

    return results