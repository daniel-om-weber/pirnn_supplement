#!/usr/bin/env python3
"""Split dataset into train/valid/test folders for tsfast."""

from pathlib import Path
import shutil
import numpy as np

# Configuration
DATASET_DIR = Path("dataset")
TRAIN_RATIO = 0.7
VALID_RATIO = 0.15
TEST_RATIO = 0.15

def split_dataset():
    """Split HDF5 files into train/valid/test folders."""
    # Create subdirectories
    train_dir = DATASET_DIR / "train"
    valid_dir = DATASET_DIR / "valid"
    test_dir = DATASET_DIR / "test"
    
    train_dir.mkdir(exist_ok=True)
    valid_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)
    
    # Get all HDF5 files in root dataset directory
    all_files = sorted([f for f in DATASET_DIR.glob("*.h5")])
    
    if not all_files:
        print("No HDF5 files found in dataset directory!")
        return
    
    # Shuffle and split
    np.random.seed(42)
    indices = np.random.permutation(len(all_files))
    
    n_train = int(len(all_files) * TRAIN_RATIO)
    n_valid = int(len(all_files) * VALID_RATIO)
    
    train_idx = indices[:n_train]
    valid_idx = indices[n_train:n_train + n_valid]
    test_idx = indices[n_train + n_valid:]
    
    # Copy files to respective directories
    for idx in train_idx:
        src = all_files[idx]
        dst = train_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
    
    for idx in valid_idx:
        src = all_files[idx]
        dst = valid_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
    
    for idx in test_idx:
        src = all_files[idx]
        dst = test_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
    
    print(f"Dataset split complete:")
    print(f"  Train: {len(train_idx)} files in {train_dir}")
    print(f"  Valid: {len(valid_idx)} files in {valid_dir}")
    print(f"  Test:  {len(test_idx)} files in {test_dir}")
    print(f"  Total: {len(all_files)} files")

if __name__ == "__main__":
    split_dataset()

