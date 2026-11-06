"""
Granular Computing Deepfake Detection
Module: src/sample_kaggle_subset.py

Fast, in-memory selective extractor for the 190k-image Kaggle Deepfake dataset (archive.zip / Dataset folder).
Does NOT extract the full archive. Uses zipfile.ZipFile to selectively read, resize (128x128), and save:
  - 1,000 Real face images into data/real/
  - 1,000 Fake face images into data/fake/
  - 25 unseen Real images into kaggle_test_samples/real/
  - 25 unseen Fake images into kaggle_test_samples/fake/
"""

import os
import io
import sys
import glob
import time
import random
import zipfile
from pathlib import Path
from typing import Optional, List, Tuple
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def find_dataset_source() -> Tuple[Optional[Path], bool]:
    """
    Locates archive.zip or an extracted Dataset directory.
    Returns: (path, is_zip_file)
    """
    candidates = [
        Path.home() / "Downloads" / "archive.zip",
        PROJECT_ROOT / "archive.zip",
        Path.home() / "Downloads" / "Dataset",
        PROJECT_ROOT / "Dataset",
    ]

    for c in candidates:
        if c.is_file() and c.suffix.lower() == ".zip":
            return c, True
        if c.is_dir():
            return c, False

    # Check for any archive*.zip in Downloads
    dl_archives = list((Path.home() / "Downloads").glob("*archive*.zip"))
    if dl_archives:
        return dl_archives[0], True

    return None, False


def clear_directory(folder: Path):
    """Safely clears existing image files from a directory."""
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)
        return

    for item in folder.iterdir():
        if item.is_file() and item.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            try:
                item.unlink()
            except Exception:
                pass


def sample_and_extract_from_zip(
    zip_path: Path,
    n_train_real: int = 1000,
    n_train_fake: int = 1000,
    n_test_real: int = 25,
    n_test_fake: int = 25,
    seed: int = 42,
):
    random.seed(seed)
    t0 = time.time()

    real_dir = PROJECT_ROOT / "data" / "real"
    fake_dir = PROJECT_ROOT / "data" / "fake"
    test_real_dir = PROJECT_ROOT / "kaggle_test_samples" / "real"
    test_fake_dir = PROJECT_ROOT / "kaggle_test_samples" / "fake"

    print(f"[1/4] Clearing old image files from {real_dir} and {fake_dir}...")
    clear_directory(real_dir)
    clear_directory(fake_dir)
    clear_directory(test_real_dir)
    clear_directory(test_fake_dir)

    print(f"[2/4] Opening archive: {zip_path} ({zip_path.stat().st_size / (1024**3):.2f} GB)...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        all_names = zf.namelist()

        # Categorize paths inside the zip archive
        train_real_candidates: List[str] = []
        train_fake_candidates: List[str] = []
        test_real_candidates: List[str] = []
        test_fake_candidates: List[str] = []

        valid_exts = {".jpg", ".jpeg", ".png"}

        for name in all_names:
            if name.endswith("/"):
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext not in valid_exts:
                continue

            lower_name = name.lower()
            if "train" in lower_name:
                if "real" in lower_name:
                    train_real_candidates.append(name)
                elif "fake" in lower_name:
                    train_fake_candidates.append(name)
            elif "test" in lower_name:
                if "real" in lower_name:
                    test_real_candidates.append(name)
                elif "fake" in lower_name:
                    test_fake_candidates.append(name)

        print(f"  Found in zip: Train Real={len(train_real_candidates):,}, Train Fake={len(train_fake_candidates):,}, "
              f"Test Real={len(test_real_candidates):,}, Test Fake={len(test_fake_candidates):,}")

        # Random sampling
        sample_tr_real = random.sample(train_real_candidates, min(n_train_real, len(train_real_candidates)))
        sample_tr_fake = random.sample(train_fake_candidates, min(n_train_fake, len(train_fake_candidates)))
        sample_te_real = random.sample(test_real_candidates, min(n_test_real, len(test_real_candidates)))
        sample_te_fake = random.sample(test_fake_candidates, min(n_test_fake, len(test_fake_candidates)))

        print(f"[3/4] Extracting & resizing to 128x128 ({len(sample_tr_real) + len(sample_tr_fake) + len(sample_te_real) + len(sample_te_fake):,} images)...")

        # Extract Train Real
        for i, name in enumerate(sample_tr_real):
            raw = zf.read(name)
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            if img.size != (128, 128):
                img = img.resize((128, 128), Image.Resampling.BILINEAR)
            img.save(real_dir / f"real_vid{i:04d}_f0.jpg", quality=95)

        # Extract Train Fake
        for i, name in enumerate(sample_tr_fake):
            raw = zf.read(name)
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            if img.size != (128, 128):
                img = img.resize((128, 128), Image.Resampling.BILINEAR)
            img.save(fake_dir / f"fake_vid{i:04d}_f0.jpg", quality=95)

        # Extract Test Real
        for i, name in enumerate(sample_te_real):
            raw = zf.read(name)
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            if img.size != (128, 128):
                img = img.resize((128, 128), Image.Resampling.BILINEAR)
            img.save(test_real_dir / f"test_real_{i:02d}.jpg", quality=95)

        # Extract Test Fake
        for i, name in enumerate(sample_te_fake):
            raw = zf.read(name)
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            if img.size != (128, 128):
                img = img.resize((128, 128), Image.Resampling.BILINEAR)
            img.save(test_fake_dir / f"test_fake_{i:02d}.jpg", quality=95)

    elapsed = time.time() - t0
    print("[4/4] Extraction complete!")
    print("=" * 65)
    print(f"  Exact Extracted Counts:")
    print(f"  - data/real/                : {len(list(real_dir.glob('*.jpg'))):,} images")
    print(f"  - data/fake/                : {len(list(fake_dir.glob('*.jpg'))):,} images")
    print(f"  - kaggle_test_samples/real/ : {len(list(test_real_dir.glob('*.jpg'))):,} images")
    print(f"  - kaggle_test_samples/fake/ : {len(list(test_fake_dir.glob('*.jpg'))):,} images")
    print(f"  Time Elapsed                : {elapsed:.2f} seconds")
    print("=" * 65)


def sample_and_extract_from_dir(
    dataset_dir: Path,
    n_train_real: int = 1000,
    n_train_fake: int = 1000,
    n_test_real: int = 25,
    n_test_fake: int = 25,
    seed: int = 42,
):
    random.seed(seed)
    t0 = time.time()

    real_dir = PROJECT_ROOT / "data" / "real"
    fake_dir = PROJECT_ROOT / "data" / "fake"
    test_real_dir = PROJECT_ROOT / "kaggle_test_samples" / "real"
    test_fake_dir = PROJECT_ROOT / "kaggle_test_samples" / "fake"

    print(f"[1/4] Clearing old image files...")
    clear_directory(real_dir)
    clear_directory(fake_dir)
    clear_directory(test_real_dir)
    clear_directory(test_fake_dir)

    print(f"[2/4] Scanning directory: {dataset_dir}...")
    valid_exts = {".jpg", ".jpeg", ".png"}

    def get_files(sub: str) -> List[Path]:
        p = dataset_dir / sub
        if not p.exists():
            # Try case-insensitive search
            for child in dataset_dir.iterdir():
                if child.name.lower() == sub.split("/")[0].lower():
                    sub_child = child / sub.split("/")[1]
                    if sub_child.exists():
                        p = sub_child
        return [f for f in p.glob("*.*") if f.suffix.lower() in valid_exts]

    tr_real = get_files("Train/Real")
    tr_fake = get_files("Train/Fake")
    te_real = get_files("Test/Real")
    te_fake = get_files("Test/Fake")

    sample_tr_real = random.sample(tr_real, min(n_train_real, len(tr_real)))
    sample_tr_fake = random.sample(tr_fake, min(n_train_fake, len(tr_fake)))
    sample_te_real = random.sample(te_real, min(n_test_real, len(te_real)))
    sample_te_fake = random.sample(te_fake, min(n_test_fake, len(te_fake)))

    print(f"[3/4] Copying & resizing to 128x128...")
    for i, p in enumerate(sample_tr_real):
        img = Image.open(p).convert("RGB")
        if img.size != (128, 128):
            img = img.resize((128, 128), Image.Resampling.BILINEAR)
        img.save(real_dir / f"real_vid{i:04d}_f0.jpg", quality=95)

    for i, p in enumerate(sample_tr_fake):
        img = Image.open(p).convert("RGB")
        if img.size != (128, 128):
            img = img.resize((128, 128), Image.Resampling.BILINEAR)
        img.save(fake_dir / f"fake_vid{i:04d}_f0.jpg", quality=95)

    for i, p in enumerate(sample_te_real):
        img = Image.open(p).convert("RGB")
        if img.size != (128, 128):
            img = img.resize((128, 128), Image.Resampling.BILINEAR)
        img.save(test_real_dir / f"test_real_{i:02d}.jpg", quality=95)

    for i, p in enumerate(sample_te_fake):
        img = Image.open(p).convert("RGB")
        if img.size != (128, 128):
            img = img.resize((128, 128), Image.Resampling.BILINEAR)
        img.save(test_fake_dir / f"test_fake_{i:02d}.jpg", quality=95)

    elapsed = time.time() - t0
    print("[4/4] Extraction complete!")
    print("=" * 65)
    print(f"  Exact Extracted Counts:")
    print(f"  - data/real/                : {len(list(real_dir.glob('*.jpg'))):,} images")
    print(f"  - data/fake/                : {len(list(fake_dir.glob('*.jpg'))):,} images")
    print(f"  - kaggle_test_samples/real/ : {len(list(test_real_dir.glob('*.jpg'))):,} images")
    print(f"  - kaggle_test_samples/fake/ : {len(list(test_fake_dir.glob('*.jpg'))):,} images")
    print(f"  Time Elapsed                : {elapsed:.2f} seconds")
    print("=" * 65)


def main():
    source_path, is_zip = find_dataset_source()
    if source_path is None:
        print("[Error] Could not find archive.zip or Dataset folder in ~/Downloads or current workspace.")
        sys.exit(1)

    print(f"[Found Dataset Source] {source_path} (is_zip={is_zip})")
    if is_zip:
        sample_and_extract_from_zip(source_path)
    else:
        sample_and_extract_from_dir(source_path)


if __name__ == "__main__":
    main()
