"""
Granular Computing-Based Multi-Level Feature Analysis for Robust Deepfake Detection
Module: src/dataset.py

Shared preprocessing, dataset loading, and dataloaders for the 2,000-image Kaggle dataset.
Key Capabilities:
1. preprocess_face(img_rgb, target_size=160): Shared preprocessing pipeline for dataset.py and app.py.
   Avoids double-cropping: does NOT run Haar cascade if image dimensions are <= 256px
   or if detected face box occupies < 55% of the frame; resizes directly to 160x160.
2. Group-Aware Stratified Train/Val/Test Split (70/15/15) with seed=42 ensuring zero data leakage.
3. Strictly [0.0, 1.0] RGB physical pixel units.
"""

import os
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union

import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def get_face_cascade() -> Optional[cv2.CascadeClassifier]:
    """Locates and initializes the OpenCV Haar Cascade for frontal face detection."""
    candidates = [
        os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"),
        str(Path(__file__).resolve().parent / "haarcascade_frontalface_default.xml"),
    ]
    for p in candidates:
        if os.path.exists(p):
            casc = cv2.CascadeClassifier(p)
            if not casc.empty():
                return casc
    return None


def preprocess_face(
    image_input: Union[Image.Image, np.ndarray],
    target_size: int = 160,
) -> Tuple[np.ndarray, torch.Tensor]:
    """
    Unified face preprocessing function used identically by dataset.py and app.py.
    Rules:
      - If image width or height is <= 256px (e.g. Kaggle pre-cropped faces), skip Haar cascade
        to prevent double-cropping into the nose, and resize directly to (target_size, target_size).
      - If image is larger than 256px, only crop if detected face area >= 55% of the image;
        otherwise resize directly.
      - Returns:
          img_np: uint8 RGB numpy array (target_size, target_size, 3) in [0, 255]
          tensor: float32 torch.Tensor (1, 3, target_size, target_size) in [0.0, 1.0]
    """
    if isinstance(image_input, Image.Image):
        img_np = np.array(image_input.convert("RGB"), dtype=np.uint8)
    else:
        img_np = image_input.copy()

    h, w = img_np.shape[:2]

    # Only attempt face detection if image is larger than 256px
    if h > 256 and w > 256:
        cascade = get_face_cascade()
        if cascade is not None:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            faces = cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=4, minSize=(60, 60))
            if len(faces) > 0:
                fx, fy, fw, fh = max(faces, key=lambda b: b[2] * b[3])
                face_area_ratio = (fw * fh) / (w * h)
                if face_area_ratio >= 0.55:
                    dx, dy = int(fw * 0.12), int(fh * 0.12)
                    x1 = max(0, fx - dx)
                    y1 = max(0, fy - dy)
                    x2 = min(w, fx + fw + dx)
                    y2 = min(h, fy + fh + dy)
                    cropped = img_np[y1:y2, x1:x2]
                    if cropped.size > 0:
                        img_np = cropped

    # Resize directly to target_size (160x160)
    if img_np.shape[:2] != (target_size, target_size):
        interp = cv2.INTER_AREA if (h > target_size and w > target_size) else cv2.INTER_LINEAR
        img_np = cv2.resize(img_np, (target_size, target_size), interpolation=interp)

    # Convert to normalized float32 tensor in [0.0, 1.0]
    tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float() / 255.0
    tensor = torch.clamp(tensor, 0.0, 1.0).unsqueeze(0)  # Shape: (1, 3, 160, 160)

    return img_np, tensor


def denormalize_tensor(tensor: torch.Tensor) -> np.ndarray:
    """Converts a (3, H, W) or (1, 3, H, W) tensor in [0.0, 1.0] back to uint8 RGB (H, W, 3)."""
    t = tensor.clone().detach().cpu()
    if t.ndim == 4:
        t = t.squeeze(0)
    t = torch.clamp(t, 0.0, 1.0)
    img_np = (t.numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
    return img_np


class GranularFaceDataset(Dataset):
    """
    PyTorch Dataset loading 160x160 face crops directly in [0.0, 1.0] RGB space.
    """

    def __init__(
        self,
        samples: List[Tuple[str, int]],
        target_size: int = 160,
        transform: Optional[transforms.Compose] = None,
        return_raw: bool = False,
    ):
        self.samples = samples
        self.target_size = target_size
        self.transform = transform
        self.return_raw = return_raw

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise IOError(f"Error loading image at {img_path}: {e}")

        # Shared preprocessor
        img_np, tensor_1 = preprocess_face(image, target_size=self.target_size)
        tensor = tensor_1.squeeze(0)  # (3, 160, 160)

        # Apply spatial data augmentations during training if supplied
        if self.transform is not None:
            pil_160 = Image.fromarray(img_np)
            tensor = self.transform(pil_160)
            tensor = torch.clamp(tensor, 0.0, 1.0)

        label_tensor = torch.tensor(label, dtype=torch.long)

        if self.return_raw:
            return tensor, label_tensor, img_np
        return tensor, label_tensor


def get_dataloaders(
    data_dir: str = "data",
    batch_size: int = 32,
    target_size: int = 160,
    split_ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    num_workers: int = 0,
    random_state: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Prepares stratified Train (70%), Validation (15%), and Test (15%) DataLoaders
    from data/real and data/fake (all 2,000 Kaggle images) at 160x160 resolution.
    """
    base_path = Path(data_dir)
    real_paths = sorted([str(p) for p in (base_path / "real").iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS])
    fake_paths = sorted([str(p) for p in (base_path / "fake").iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS])

    print(f"[Dataset Loader] Found {len(real_paths)} Real and {len(fake_paths)} Fake images in {base_path}.")

    def extract_group_id(path_str: str) -> str:
        stem = Path(path_str).stem
        parts = stem.split("_f")
        return parts[0] if len(parts) > 1 else stem

    def group_aware_split(paths: List[str], label: int):
        group_to_paths: Dict[str, List[str]] = {}
        for p in paths:
            gid = extract_group_id(p)
            group_to_paths.setdefault(gid, []).append(p)

        unique_groups = sorted(list(group_to_paths.keys()))
        val_test_share = split_ratios[1] + split_ratios[2]

        train_groups, temp_groups = train_test_split(
            unique_groups,
            test_size=val_test_share,
            random_state=random_state,
        )

        val_sub_share = split_ratios[1] / val_test_share
        val_groups, test_groups = train_test_split(
            temp_groups,
            test_size=(1.0 - val_sub_share),
            random_state=random_state,
        )

        train_s = [(p, label) for g in train_groups for p in group_to_paths[g]]
        val_s = [(p, label) for g in val_groups for p in group_to_paths[g]]
        test_s = [(p, label) for g in test_groups for p in group_to_paths[g]]

        return train_s, val_s, test_s

    real_train, real_val, real_test = group_aware_split(real_paths, label=0)
    fake_train, fake_val, fake_test = group_aware_split(fake_paths, label=1)

    train_samples = real_train + fake_train
    val_samples = real_val + fake_val
    test_samples = real_test + fake_test

    # Spatial augmentations during training
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=4),
        transforms.ToTensor(),
    ])

    train_dataset = GranularFaceDataset(train_samples, target_size=target_size, transform=train_transform)
    val_dataset = GranularFaceDataset(val_samples, target_size=target_size, transform=None)
    test_dataset = GranularFaceDataset(test_samples, target_size=target_size, transform=None)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(
        f"[DataLoaders Ready] Total: {len(real_paths) + len(fake_paths)} images | "
        f"Train: {len(train_dataset)} ({len(train_loader)} batches) | "
        f"Val: {len(val_dataset)} ({len(val_loader)} batches) | "
        f"Test: {len(test_dataset)} ({len(test_loader)} batches) | "
        f"Target Resolution: {target_size}x{target_size}"
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_l, val_l, test_l = get_dataloaders(batch_size=32, target_size=160)
    for bx, by in train_l:
        print(f"Sample Batch: X={bx.shape}, Y={by.shape}, Range=[{bx.min():.3f}, {bx.max():.3f}]")
        break
