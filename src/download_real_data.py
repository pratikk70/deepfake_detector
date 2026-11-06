"""
Granular Computing Deepfake Detection
Module: src/download_real_data.py

Downloads authentic and deepfake face images from real datasets:
Primary source: HuggingFace dataset 'JamieWithofs/Deepfake-and-real-images-validation'
(streaming=True, 2041 examples, 600x600 real and fake face images).
Crops/resizes each face to 128x128 and saves:
  - 250 Real images into data/real/real_vid{i}_f0.jpg
  - 250 Fake images into data/fake/fake_vid{i}_f0.jpg
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def get_face_cascade():
    candidates = [
        os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"),
        str(PROJECT_ROOT / "src" / "haarcascade_frontalface_default.xml"),
    ]
    for c in candidates:
        if os.path.exists(c):
            casc = cv2.CascadeClassifier(c)
            if not casc.empty():
                return casc
    return None


def crop_and_resize_face(pil_img: Image.Image, cascade, target_size=(128, 128)) -> Image.Image:
    """Detects face and crops with 15% margin, or resizes directly if already cropped."""
    img_np = np.array(pil_img.convert("RGB"))
    h, w = img_np.shape[:2]

    if cascade is not None:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=4, minSize=(60, 60))
        if len(faces) > 0:
            # Pick largest detected face
            fx, fy, fw, fh = max(faces, key=lambda b: b[2] * b[3])
            dx, dy = int(fw * 0.15), int(fh * 0.15)
            x1 = max(0, fx - dx)
            y1 = max(0, fy - dy)
            x2 = min(w, fx + fw + dx)
            y2 = min(h, fy + fh + dy)
            cropped_np = img_np[y1:y2, x1:x2]
            if cropped_np.size > 0:
                resized = cv2.resize(cropped_np, target_size, interpolation=cv2.INTER_AREA)
                return Image.fromarray(resized)

    # Fallback: direct high quality resize
    return pil_img.resize(target_size, Image.Resampling.BILINEAR)


def download_real_and_fake_faces(
    target_real: int = 250,
    target_fake: int = 250,
    output_dir: str = "data",
):
    from datasets import load_dataset

    out_path = Path(output_dir)
    real_dir = out_path / "real"
    fake_dir = out_path / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    cascade = get_face_cascade()
    real_count = len(list(real_dir.glob("*.jpg")))
    fake_count = len(list(fake_dir.glob("*.jpg")))

    print("=" * 65)
    print("  Downloading Real & Deepfake Face Dataset")
    print(f"  Target: {target_real} Real, {target_fake} Fake images (128x128)")
    print(f"  Current: {real_count} Real, {fake_count} Fake")
    print("=" * 65)

    if real_count >= target_real and fake_count >= target_fake:
        print("[Dataset] Already populated with required real and fake images!")
        return

    dataset_name = "JamieWithofs/Deepfake-and-real-images-validation"
    print(f"[HuggingFace] Connecting to stream: {dataset_name}...")

    try:
        ds = load_dataset(dataset_name, split="validation", streaming=True)
    except Exception as e:
        print(f"[Error] Failed to load {dataset_name}: {e}")
        print("[HuggingFace] Trying fallback: Hemg/deepfake-and-real-images...")
        ds = load_dataset("Hemg/deepfake-and-real-images", split="test", streaming=True)

    stream_idx = 0
    for sample in ds:
        img_raw = sample.get("image")
        label = sample.get("label")  # 0 = Fake, 1 = Real

        if img_raw is None or label is None:
            continue

        if not isinstance(img_raw, Image.Image):
            continue

        # In JamieWithofs dataset: 0 = Fake, 1 = Real
        is_fake = (label == 0)

        if is_fake and fake_count < target_fake:
            face_128 = crop_and_resize_face(img_raw, cascade, target_size=(128, 128))
            save_name = f"fake_vid{fake_count:04d}_f0.jpg"
            face_128.save(fake_dir / save_name, quality=95)
            fake_count += 1
            if fake_count % 25 == 0 or fake_count == target_fake:
                print(f"  [Progress] Fake faces: {fake_count}/{target_fake}")

        elif (not is_fake) and real_count < target_real:
            face_128 = crop_and_resize_face(img_raw, cascade, target_size=(128, 128))
            save_name = f"real_vid{real_count:04d}_f0.jpg"
            face_128.save(real_dir / save_name, quality=95)
            real_count += 1
            if real_count % 25 == 0 or real_count == target_real:
                print(f"  [Progress] Real faces: {real_count}/{target_real}")

        stream_idx += 1
        if real_count >= target_real and fake_count >= target_fake:
            break

    print(f"\n[Download Complete] Successfully saved {real_count} Real and {fake_count} Fake faces in {out_path}/")


if __name__ == "__main__":
    download_real_and_fake_faces(target_real=250, target_fake=250)
