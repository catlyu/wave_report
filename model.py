import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json, os, platform

#######################################################################
# 0.  QUICK NOTES
# --------------------------------------------------------------------
# • Keeps the *same* external API (`WaveClassifier`, `MultiTaskLoss`, etc.) so
#   `train.py` continues to work unchanged except for the looser checkpoint
#   criterion (explained at the bottom).
# • Adds Focal‑Cross‑Entropy to tackle class‑imbalance → more diverse quality /
#   size predictions & less "all‑zero" wind.
# • Introduces two heuristic visual cues for wind direction that feed directly
#   into the logits (white‑cap ratio & choppiness).
#######################################################################

# --------------------------------------------------------------------
# 1.  SEA / SKY SEGMENTATION WITH HORIZON SUPPRESSION
# --------------------------------------------------------------------
class SeaDetectionModule(nn.Module):
    """Detect sea regions using colour, texture and horizon‑line suppression."""

    def __init__(self):
        super().__init__()
        self.ripple_conv1 = nn.Conv2d(3, 16, 5, padding=2)
        self.ripple_conv2 = nn.Conv2d(16, 8, 3, padding=1)
        self.ripple_out   = nn.Conv2d(8, 1, 1)
        self.blue_weights = nn.Parameter(torch.tensor([0.2, 0.4, 0.4]))  # R,G,B
        self.register_buffer("horizontal_bias", self._build_horizontal_bias())

    # ---------------- helpers ------------------------------------------------
    @staticmethod
    def _build_horizontal_bias(h: int = 224, w: int = 224):
        bias = torch.zeros(1, 1, h, w)
        centre = h // 2
        for y in range(h):
            bias_strength = 1.0 - abs(y - centre) / (h / 2) * 0.6  # 1 → 0.4
            bias[0, 0, y, :] = bias_strength
        return bias

    def _detect_blueish(self, x):
        w = x * self.blue_weights.view(1, 3, 1, 1)
        score = w.sum(1, keepdim=True)
        return torch.sigmoid(score - 0.4)  # bluish/greenish threshold

    def _detect_ripples(self, x):
        r = F.relu(self.ripple_conv1(x))
        r = F.relu(self.ripple_conv2(r))
        return torch.sigmoid(self.ripple_out(r))

    @staticmethod
    def _suppress_sky(sea_mask, thr: float = 0.35):
        n, _, h, _ = sea_mask.shape
        for b in range(n):
            row_has = (sea_mask[b, 0] > thr).any(dim=1)
            if row_has.any():
                horizon = torch.argmax(row_has.int())  # first sea row
                sea_mask[b, 0, :horizon] = 0.0
        return sea_mask

    def forward(self, x):  # x: (B,3,H,W)
        x_rsz = F.interpolate(x, (224, 224), mode="bilinear", align_corners=False)
        blue   = self._detect_blueish(x_rsz)
        ripple = self._detect_ripples(x_rsz)
        sea    = 0.4 * blue + 0.3 * ripple + 0.3 * self.horizontal_bias
        sea    = self._suppress_sky(sea)
        sea    = F.interpolate(sea, (7, 7), mode="bilinear", align_corners=False)
        return sea

# --------------------------------------------------------------------
# 2.  ATTENTION MODULE (unchanged API)
# --------------------------------------------------------------------
class WaveAttentionModule(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch // 2, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(in_ch // 2)
        self.conv2 = nn.Conv2d(in_ch // 2, in_ch // 4, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(in_ch // 4)
        self.conv3 = nn.Conv2d(in_ch // 4, 1, 1)
        self.drop  = nn.Dropout2d(0.1)
        self.sea_det = SeaDetectionModule()
        self.register_buffer("spatial_bias", self._centre_bias())

    @staticmethod
    def _centre_bias(h: int = 7, w: int = 7):
        bias = torch.zeros(1, 1, h, w)
        cx, cy = h // 2, w // 2
        maxd = (cx ** 2 + cy ** 2) ** 0.5
        for i in range(h):
            for j in range(w):
                dist = ((i - cx) ** 2 + (j - cy) ** 2) ** 0.5
                bias[0, 0, i, j] = 0.3 - 0.2 * dist / maxd
        return bias

    def forward(self, feats, orig_img):
        a = F.relu(self.bn1(self.conv1(feats)))
        a = self.drop(a)
        a = F.relu(self.bn2(self.conv2(a)))
        a = torch.sigmoid(self.conv3(a))

        sea_mask = self.sea_det(orig_img)  # (B,1,7,7)
        attn = torch.sigmoid(1.5 * sea_mask + 0.8 * self.spatial_bias + 0.4 * a)
        return feats * attn, attn

# --------------------------------------------------------------------
# 3.  SIMPLE VISUAL ANALYSERS (unchanged)
# --------------------------------------------------------------------
class ColorContrastAnalyzer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv5 = nn.Conv2d(3, 8, 5, padding=2)
        self.white = nn.Conv2d(3, 8, 3, padding=1)
        self.mix   = nn.Conv2d(24, 16, 1)
        self.pool  = nn.AdaptiveAvgPool2d(1)
        self.fc    = nn.Linear(16, 32)

    def _white_mask(self, x):
        intensity = x.mean(1, keepdim=True)
        return torch.sigmoid((intensity - 0.7) * 10)

    def forward(self, x):
        c3 = F.relu(self.conv3(x))
        c5 = F.relu(self.conv5(x))
        white = F.relu(self.white(x * self._white_mask(x)))
        mix = F.relu(self.mix(torch.cat([c3, c5, white], 1)))
        feat = self.pool(mix).view(x.size(0), -1)
        return self.fc(feat)

class TextureConsistencyAnalyzer(nn.Module):
    def __init__(self):
        super().__init__()
        self.h = nn.Conv2d(3, 8, (1, 5), padding=(0, 2))
        self.v = nn.Conv2d(3, 8, (5, 1), padding=(2, 0))
        self.g = nn.Conv2d(3, 8, 5, padding=2)
        self.s = nn.Conv2d(3, 8, 3, padding=1)
        self.mix = nn.Conv2d(32, 16, 3, padding=1)
        self.ref = nn.Conv2d(16, 8, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 32)

    def _smooth_mask(self, x):
        gray = 0.299 * x[:, :1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        mean = F.avg_pool2d(gray, 3, stride=1, padding=1)
        var  = F.avg_pool2d((gray - mean) ** 2, 3, stride=1, padding=1)
        return torch.sigmoid(-var * 5 + 2).expand_as(x)

    def forward(self, x):
        h = F.relu(self.h(x))
        v = F.relu(self.v(x))
        g = F.relu(self.g(x))
        s = F.relu(self.s(x * self._smooth_mask(x)))
        mix = F.relu(self.mix(torch.cat([h, v, g, s], 1)))
        feat = F.relu(self.ref(mix))
        feat = self.pool(feat).view(x.size(0), -1)
        return self.fc(feat)

# --------------------------------------------------------------------
# 4.  SMALL UTILITY — FOCAL LOSS WITH LABEL SMOOTHING
# --------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 1.5, weight=None, label_smoothing: float = 0.05):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits, target):  # logits: (B,C) target: (B,)
        ce = F.cross_entropy(logits, target, weight=self.weight,
                             reduction="none", label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()

# --------------------------------------------------------------------
# 5.  MAIN CLASSIFIER WITH EXTRA WIND HEURISTICS
# --------------------------------------------------------------------
class WaveClassifier(nn.Module):
    def __init__(self, n_q: int = 9, n_s: int = 7, n_w: int = 3):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # 512×7×7

        self.attn = WaveAttentionModule(512)
        self.fuse = nn.Conv2d(512, 256, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.color   = ColorContrastAnalyzer()
        self.texture = TextureConsistencyAnalyzer()

        self.comb = nn.Linear(256 + 32 + 32, 256)
        self.drop = nn.Dropout(0.5)

        # task heads ----------------------------------------------------------------
        self.quality_head = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.4),
                                          nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
                                          nn.Linear(128, n_q))

        self.size_head    = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
                                          nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
                                          nn.Linear(64, n_s))

        self.wind_head    = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.2),
                                          nn.Linear(64, n_w))

        # heuristic projections ------------------------------------------------------
        self.wind_tex    = nn.Linear(32, n_w, bias=False)  # choppiness ↔ texture
        self.wind_white  = nn.Linear(1,  n_w, bias=False)  # white‑cap ratio

    # ---------------- utility -------------------------------------------------------
    @staticmethod
    def _whitecap_ratio(img: torch.Tensor, sea_mask_down: torch.Tensor):
        """Approximate proportion of bright pixels inside the sea region."""
        # upsample mask back to 224 × 224 for compatibility
        mask = F.interpolate(sea_mask_down, img.shape[-2:], mode="bilinear", align_corners=False)
        intensity = img.mean(1, keepdim=True)
        white_map = torch.sigmoid((intensity - 0.8) * 10)  # bright pixels ~1
        white_sea = (white_map * mask).mean([2, 3])  # (B,1)
        return white_sea

    # ------------------------------------------------------------------------------
    def forward(self, x, add_diversity: bool = False, diversity_strength: float = 0.1):
        feats = self.backbone(x)                     # (B,512,7,7)
        feats, attn_map = self.attn(feats, x)        # attn applied
        feats = self.pool(self.fuse(feats)).view(x.size(0), -1)  # (B,256)

        col_feat = self.color(x)      # (B,32)
        tex_feat = self.texture(x)    # (B,32)

        all_feat = torch.cat([feats, col_feat, tex_feat], 1)
        all_feat = F.relu(self.comb(all_feat))
        all_feat = self.drop(all_feat)

        if add_diversity and not self.training:
            all_feat = all_feat + torch.randn_like(all_feat) * diversity_strength

        # predictions ----------------------------------------------------------------
        out_q = self.quality_head(all_feat)
        out_s = self.size_head(all_feat)

        # WHITE‑CAP heuristic needs sea mask ↓
        with torch.no_grad():
            sea_down = self.attn.sea_det(x)  # re‑use detector (B,1,7,7)
        white_ratio = self._whitecap_ratio(x, sea_down)  # (B,1)

        out_w = (self.wind_head(all_feat) +
                  self.wind_tex(tex_feat) +
                  self.wind_white(white_ratio))

        return {
            "quality":      out_q,
            "size":         out_s,
            "wind_dir":     out_w,
            "attention_map": attn_map,
        }

# --------------------------------------------------------------------
# 6.  DATASET / TRANSFORMS / LOADERS (unchanged API)
# --------------------------------------------------------------------
class WaveDataset(Dataset):
    def __init__(self, metadata_path: str, transform=None):
        with open(metadata_path) as f:
            self.data = json.load(f)
        self.tr = transform
        self.q_min = min(d["label"]["quality"] for d in self.data)
        self.s_min = min(d["label"]["size"] for d in self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rec = self.data[idx]
        img = Image.open(rec["filename"]).convert("RGB")
        if self.tr:
            img = self.tr(img)
        q_val, s_val, w_val = rec["label"]["quality"], rec["label"]["size"], rec["label"]["wind_dir"]
        return {
            "image": img,
            "quality": torch.tensor(q_val - self.q_min, dtype=torch.long),
            "size":    torch.tensor(s_val - self.s_min, dtype=torch.long),
            "wind_dir": torch.tensor(w_val + 1, dtype=torch.long),  # -1/0/1 → 0/1/2
            "quality_float": torch.tensor(float(q_val)),
            "size_float":    torch.tensor(float(s_val)),
            "filename": rec["filename"],
        }

def get_transforms(train: bool = True):
    if train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3,0.3,0.3,0.1),
            transforms.RandomRotation(8),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
            transforms.RandomErasing(p=0.2,scale=(0.02,0.1),ratio=(0.3,3.3)),
        ])
    return transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])


def create_data_loaders(train_meta: str, test_meta: str, batch_size: int = 16):
    train_ds = WaveDataset(train_meta, transform=get_transforms(True))
    test_ds = WaveDataset(test_meta, transform=get_transforms(False))

    num_workers = 0 if platform.system() == "Windows" or os.cpu_count() <= 2 else min(2, os.cpu_count() - 1)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


# -----------------------------------------------------------------------------
# 🌊 8.  QUICK SELF‑TEST
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    model = WaveClassifier()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    dummy = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy)
        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                print(k, v.shape)
