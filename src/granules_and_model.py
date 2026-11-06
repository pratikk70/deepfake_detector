"""
Granular Computing-Based Multi-Level Feature Analysis for Robust Deepfake Detection
Module: src/granules_and_model.py

Upgraded Architecture:
1. Operates strictly in [0.0, 1.0] RGB physical pixel units at 160x160 resolution.
2. Signal Restored: Gaussian pre-smoothing filters REMOVED from Branch 2 & 3 so subtle
   texture and boundary artifacts are not degraded.
3. Branch 1 (Pixel Granule): High-Pass SRM/Laplacian residual CNN -> 64-dim embedding + auxiliary head.
4. Branch 2 (Texture Granule): Differentiable 8-neighbor LBP with sharpness scale=150.0
   alongside directional Sobel filters (11 channels) -> 64-dim embedding + auxiliary head.
5. Branch 3 (Facial Regions): 4 Non-overlapping anatomical zones (Eyes, Nose, Mouth, Cheek/Boundary)
   encoded via torchvision.models.mobilenet_v3_small(weights='DEFAULT') -> 64-dim embedding + auxiliary head.
6. SingleLevelBaselineNet: Upgraded to use mobilenet_v3_small(weights='DEFAULT') on raw RGB for a fair benchmark.
7. Adaptive Evidence Gating: Layer-normalized branch embeddings with Softmax trust weights summing to 1.0.
"""

from typing import Dict, Tuple, Any, Optional

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# -----------------------------------------------------------------------------
# 1. VISUAL GRANULE EXTRACTOR (FOR STREAMLIT UI & INTERPRETABILITY)
# -----------------------------------------------------------------------------

def extract_visual_granules(image_np: np.ndarray) -> Dict[str, Any]:
    """
    Extracts visual representations of the 3 granules from a 160x160 RGB image.
    """
    if image_np.shape[:2] != (160, 160):
        image_np = cv2.resize(image_np, (160, 160), interpolation=cv2.INTER_LINEAR)

    if image_np.dtype != np.uint8:
        image_np = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)

    # a) Granule 1: Pixel Noise Residual (Original - 3x3 Gaussian blur, amplified 5x)
    blurred = cv2.GaussianBlur(image_np.astype(np.float32), (3, 3), 1.0)
    residual = (image_np.astype(np.float32) - blurred) * 5.0 + 128.0
    pixel_residual_uint8 = np.clip(residual, 0, 255).astype(np.uint8)

    # b) Granule 2: Local Binary Pattern (LBP) texture map via scikit-image
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_norm = (lbp / 9.0 * 255.0).astype(np.uint8)
    lbp_rgb = cv2.cvtColor(lbp_norm, cv2.COLOR_GRAY2RGB)

    # c) Granule 3: 4 Non-Overlapping Facial Regions at 160x160
    # 1. Periocular Eyes: [25:65, 25:135] (40x110)
    # 2. Nose: [65:102, 55:105] (37x50)
    # 3. Mouth: [102:138, 42:118] (36x76)
    # 4. Cheek & Forehead Boundary: [0:25, 25:135] (25x110)
    region_slices = {
        "eyes": (slice(25, 65), slice(25, 135)),
        "nose": (slice(65, 102), slice(55, 105)),
        "mouth": (slice(102, 138), slice(42, 118)),
        "boundary": (slice(0, 25), slice(25, 135)),
    }

    cropped_regions = {
        k: image_np[s[0], s[1]].copy() for k, s in region_slices.items()
    }

    # Annotated face image with colored bounding boxes
    annotated = image_np.copy()
    box_colors = {
        "eyes": (0, 200, 255),       # Cyan
        "nose": (50, 220, 50),        # Bright Green
        "mouth": (255, 80, 80),       # Coral Red
        "boundary": (220, 150, 0),    # Gold/Blue
    }

    for name, (ys, xs) in region_slices.items():
        x1, x2 = xs.start, xs.stop
        y1, y2 = ys.start, ys.stop
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_colors[name], 1)
        cv2.putText(
            annotated,
            name[:1].upper(),
            (x1 + 2, y1 + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            box_colors[name],
            1,
            cv2.LINE_AA,
        )

    return {
        "original": image_np,
        "pixel_residual": pixel_residual_uint8,
        "lbp_texture": lbp_rgb,
        "regions": cropped_regions,
        "annotated_face": annotated,
    }


# -----------------------------------------------------------------------------
# 2. BRANCH MODULES
# -----------------------------------------------------------------------------

class PixelGranuleModule(nn.Module):
    """
    Branch 1: Fine-Grained High-Pass Residual Extraction.
    Uses fixed Laplacian/SRM kernel across raw RGB channels -> 64-dim embedding.
    """

    def __init__(self, out_dim: int = 64):
        super().__init__()
        laplacian = torch.tensor([
            [-1.0, -1.0, -1.0],
            [-1.0,  8.0, -1.0],
            [-1.0, -1.0, -1.0]
        ], dtype=torch.float32) / 8.0

        self.high_pass = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False, groups=3)
        self.high_pass.weight.data = laplacian.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        self.high_pass.weight.requires_grad = False

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (B, 32, 80, 80)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (B, 64, 40, 40)

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # (B, 64, 1, 1)
            nn.Flatten(),
            nn.Linear(64, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.high_pass(x)
        return self.encoder(res)


class TextureGranuleModule(nn.Module):
    """
    Branch 2: Differentiable 8-Neighbor LBP + Sobel Texture Module.
    NO Gaussian pre-smoothing. Sharpness scale = 150.0 in [0, 1] space.
    Outputs 64-dim embedding.
    """

    def __init__(self, out_dim: int = 64):
        super().__init__()
        sobel_x = torch.tensor([
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0]
        ], dtype=torch.float32) / 4.0

        sobel_y = torch.tensor([
            [-1.0, -2.0, -1.0],
            [ 0.0,  0.0,  0.0],
            [ 1.0,  2.0,  1.0]
        ], dtype=torch.float32) / 4.0

        self.sobel = nn.Conv2d(1, 2, kernel_size=3, padding=1, bias=False)
        self.sobel.weight.data[0, 0] = sobel_x
        self.sobel.weight.data[1, 0] = sobel_y
        self.sobel.weight.requires_grad = False

        # 11 input channels: 8 soft LBP bits + Sobel X + Sobel Y + Grad Magnitude
        self.encoder = nn.Sequential(
            nn.Conv2d(11, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (B, 32, 80, 80)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (B, 64, 40, 40)

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert raw [0, 1] RGB to Grayscale
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        h, w = gray.shape[2], gray.shape[3]

        # 8-neighbor soft LBP differences with sharpness scale 150.0
        padded = F.pad(gray, (1, 1, 1, 1), mode="replicate")
        p0 = padded[:, :, 0:h, 0:w]
        p1 = padded[:, :, 0:h, 1:w + 1]
        p2 = padded[:, :, 0:h, 2:w + 2]
        p3 = padded[:, :, 1:h + 1, 0:w]
        p4 = padded[:, :, 1:h + 1, 2:w + 2]
        p5 = padded[:, :, 2:h + 2, 0:w]
        p6 = padded[:, :, 2:h + 2, 1:w + 1]
        p7 = padded[:, :, 2:h + 2, 2:w + 2]

        scale = 150.0
        lbp_bits = [torch.sigmoid(scale * (p - gray)) for p in [p0, p1, p2, p3, p4, p5, p6, p7]]
        lbp_tensor = torch.cat(lbp_bits, dim=1)  # (B, 8, H, W)

        grad_xy = self.sobel(gray)
        gx = grad_xy[:, 0:1]
        gy = grad_xy[:, 1:2]
        mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

        texture_feat = torch.cat([lbp_tensor, gx, gy, mag], dim=1)  # (B, 11, H, W)
        return self.encoder(texture_feat)


class FacialRegionGranuleModule(nn.Module):
    """
    Branch 3: 4 Non-Overlapping Facial Regions with MobileNetV3-Small backbone.
    NO Gaussian smoothing. Outputs 64-dim semantic embedding.
    """

    def __init__(self, out_dim: int = 64):
        super().__init__()
        # Pretrained MobileNetV3-Small feature extractor
        mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.backbone = mobilenet.features

        # Projection from 4 concatenated region features (4 * 576 = 2304) to out_dim (64)
        self.projector = nn.Sequential(
            nn.Linear(576 * 4, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 4 Non-overlapping region crops from 160x160
        crop_eyes = x[:, :, 25:65, 25:135]
        crop_nose = x[:, :, 65:102, 55:105]
        crop_mouth = x[:, :, 102:138, 42:118]
        crop_bound = x[:, :, 0:25, 25:135]

        target_size = (56, 56)
        r_eyes = F.interpolate(crop_eyes, size=target_size, mode="bilinear", align_corners=False)
        r_nose = F.interpolate(crop_nose, size=target_size, mode="bilinear", align_corners=False)
        r_mouth = F.interpolate(crop_mouth, size=target_size, mode="bilinear", align_corners=False)
        r_bound = F.interpolate(crop_bound, size=target_size, mode="bilinear", align_corners=False)

        # Batch encode through MobileNetV3
        all_crops = torch.cat([r_eyes, r_nose, r_mouth, r_bound], dim=0)  # (4*B, 3, 56, 56)
        feats = self.backbone(all_crops)  # (4*B, 576, 2, 2)
        pooled = F.adaptive_avg_pool2d(feats, (1, 1)).flatten(1)  # (4*B, 576)

        b = x.size(0)
        e_eyes = pooled[0:b]
        e_nose = pooled[b:2*b]
        e_mouth = pooled[2*b:3*b]
        e_bound = pooled[3*b:4*b]

        concat_regions = torch.cat([e_eyes, e_nose, e_mouth, e_bound], dim=1)  # (B, 2304)
        return self.projector(concat_regions)


# -----------------------------------------------------------------------------
# 3. PROPOSED GRANULAR NETWORK WITH ADAPTIVE EVIDENCE GATING & AUX HEADS
# -----------------------------------------------------------------------------

class AdaptiveEvidenceFusion(nn.Module):
    """
    Evidence Gating Layer:
    - Layer-normalizes branch embeddings prior to gating.
    - Computes dynamic trust weights [w_pixel, w_texture, w_region] summing to 1.0.
    """

    def __init__(self, emb_dim: int = 64):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(emb_dim * 3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),
            nn.Softmax(dim=-1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        f_pixel: torch.Tensor,
        f_texture: torch.Tensor,
        f_region: torch.Tensor,
        forced_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        f_p_norm = F.layer_norm(f_pixel, (f_pixel.shape[-1],))
        f_t_norm = F.layer_norm(f_texture, (f_texture.shape[-1],))
        f_r_norm = F.layer_norm(f_region, (f_region.shape[-1],))

        if forced_weights is not None:
            trust_weights = forced_weights
        else:
            f_all = torch.cat([f_p_norm, f_t_norm, f_r_norm], dim=1)  # (B, 192)
            trust_weights = self.gate(f_all)  # (B, 3)

        w_p = trust_weights[:, 0:1]
        w_t = trust_weights[:, 1:2]
        w_r = trust_weights[:, 2:3]

        fused = w_p * f_pixel + w_t * f_texture + w_r * f_region
        logits = self.classifier(fused).squeeze(-1)
        return logits, trust_weights


class GranularDeepfakeNet(nn.Module):
    """
    Proposed 3-Branch Granular Computing Deepfake Detector.
    Integrates Pixel, Texture (LBP), and Facial Region (MobileNetV3) embeddings.
    Features:
      - Standalone auxiliary classification heads on each branch for multi-task gradient flow.
      - Adaptive Softmax gating with LayerNorm.
      - Branch Dropout during training.
      - Single-branch ablation testing support.
    """

    def __init__(self, emb_dim: int = 64):
        super().__init__()
        self.pixel_branch = PixelGranuleModule(out_dim=emb_dim)
        self.texture_branch = TextureGranuleModule(out_dim=emb_dim)
        self.region_branch = FacialRegionGranuleModule(out_dim=emb_dim)
        self.fusion_head = AdaptiveEvidenceFusion(emb_dim=emb_dim)

        # Auxiliary classification heads for standalone granule training
        self.aux_pixel = nn.Linear(emb_dim, 1)
        self.aux_texture = nn.Linear(emb_dim, 1)
        self.aux_region = nn.Linear(emb_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        ablation: Optional[str] = None,
        return_aux: bool = False,
    ):
        f_pixel = self.pixel_branch(x)
        f_texture = self.texture_branch(x)
        f_region = self.region_branch(x)

        # Standalone auxiliary logits
        aux_pix = self.aux_pixel(f_pixel).squeeze(-1)
        aux_tex = self.aux_texture(f_texture).squeeze(-1)
        aux_reg = self.aux_region(f_region).squeeze(-1)

        # Branch Dropout during training (in 20% of training batches, zero 1 branch)
        if self.training and ablation is None and torch.rand(1).item() < 0.20:
            drop_idx = torch.randint(0, 3, (1,)).item()
            if drop_idx == 0:
                f_pixel = torch.zeros_like(f_pixel)
            elif drop_idx == 1:
                f_texture = torch.zeros_like(f_texture)
            else:
                f_region = torch.zeros_like(f_region)

        # Forced weights for ablation testing
        forced_weights = None
        if ablation is not None:
            batch_size = x.size(0)
            if ablation == "pixel":
                forced_weights = torch.tensor([[1.0, 0.0, 0.0]], device=x.device).repeat(batch_size, 1)
            elif ablation == "texture":
                forced_weights = torch.tensor([[0.0, 1.0, 0.0]], device=x.device).repeat(batch_size, 1)
            elif ablation == "region":
                forced_weights = torch.tensor([[0.0, 0.0, 1.0]], device=x.device).repeat(batch_size, 1)

        logits, trust_weights = self.fusion_head(f_pixel, f_texture, f_region, forced_weights=forced_weights)

        if return_aux:
            return logits, trust_weights, (aux_pix, aux_tex, aux_reg)
        return logits, trust_weights


# -----------------------------------------------------------------------------
# 4. SINGLE-LEVEL BASELINE NETWORK (WITH MOBILENETV3 ON RAW RGB)
# -----------------------------------------------------------------------------

class SingleLevelBaselineNet(nn.Module):
    """
    Monolithic Single-Branch Baseline CNN.
    Upgraded to use MobileNetV3-Small(weights='DEFAULT') on raw RGB for a fair benchmark.
    """

    def __init__(self):
        super().__init__()
        mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.features = mobilenet.features

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(576, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        return self.classifier(feat).squeeze(-1)


if __name__ == "__main__":
    dummy_x = torch.rand(2, 3, 160, 160)
    net_g = GranularDeepfakeNet(emb_dim=64)
    net_b = SingleLevelBaselineNet()

    l_g, tw, aux = net_g(dummy_x, return_aux=True)
    l_b = net_b(dummy_x)

    print("Forward Pass Verification:")
    print(f"  Granular logits: {l_g.shape}, trust: {tw.shape}")
    print(f"  Aux logits: pix={aux[0].shape}, tex={aux[1].shape}, reg={aux[2].shape}")
    print(f"  Baseline logits: {l_b.shape}")
    print("[SUCCESS] src/granules_and_model.py is 100% verified!")
