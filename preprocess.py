#!/usr/bin/env python3
"""
combined_grace_pipeline.py

One-stop pipeline that:
  1) Discovers paired image/label files in a GRACE-like source tree
  2) Splits cases into train/val/test
  3) Copies files into an nnU-Net-style directory layout:
        imagesTr/, labelsTr/, imagesTs/ and labelsTs/ 
  4) Generates a dataset.json compatible with many medical-imaging pipelines

Usage (examples):
  python combined_grace_pipeline.py \
      --data /path/to/raw_grace \
      --source-folders folder1 folder2 folder3
      --verbose
"""

import argparse
import json
import os
import random
import re
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split


def main():
    p = argparse.ArgumentParser(description="Combined GRACE data prep pipeline.")
    p.add_argument("--data", type=str, required=True, help="Path to the base GRACE data directory containing source folders.")
    p.add_argument("--source-folders", type=str, nargs='+', required=True, help="List of source folder names to search within the base data directory.")
    p.add_argument("--verbose", action="store_true", help="Enable verbose output.")

    args = p.parse_args()
    base_dir = Path(args.data)
    source_folders = args.source_folders
    if not base_dir.is_dir():
        print(f"Base data directory does not exist: {base_dir}")
        return
    for folder in source_folders:
        if not (base_dir / folder).is_dir():
            print(f"Source folder does not exist: {base_dir / folder}")
            return
        
    # Step 1: Copy files
    if args.verbose:
        print("[GRACE] Copying files from source folders...")
    random.seed(42)
    images_dir = base_dir / "images"
    labels_dir = base_dir / "labels"
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)

    for folder in source_folders:
        full_folder_path = base_dir / folder
        for subfolder in full_folder_path.iterdir():
            if not subfolder.is_dir() or '_MISSING' in subfolder.name:
                continue
            try:
                numeric_id = re.search(r'sub-(\d+)', subfolder.name).group(1)
                t1_path = subfolder / "T1.nii"
                mask_path = subfolder / "T1_T1orT2_masks.nii"
                if t1_path.exists():
                    shutil.copy(t1_path, images_dir / f"{numeric_id}.nii")
                if mask_path.exists():
                    shutil.copy(mask_path, labels_dir / f"{numeric_id}.nii")
            except Exception as e:
                print(f"Error processing {subfolder}: {e}")
    if args.verbose:
        if not os.listdir(images_dir) or not os.listdir(labels_dir):
            print("[GRACE] Error: No files were copied. Please check the source folders and their structure.")
            raise RuntimeError("No files copied.")
        else:
            print("[GRACE] File copying completed.")

    # Step 2: Split data
    if args.verbose:
        print("[GRACE] Splitting data into train/test sets...")
    dest_folders = {
        "imagesTr": base_dir / "imagesTr",
        "imagesTs": base_dir / "imagesTs",
        "labelsTr": base_dir / "labelsTr",
        "labelsTs": base_dir / "labelsTs",
    }

    # Make sure destination folders exist
    for path in dest_folders.values():
        os.makedirs(path, exist_ok=True)

    imagesTr_dir = Path(dest_folders["imagesTr"])
    labelsTr_dir = Path(dest_folders["labelsTr"])
    imagesTs_dir = Path(dest_folders["imagesTs"])
    labelsTs_dir = Path(dest_folders["labelsTs"])
    
    # List all image files
    all_image_files = [f for f in os.listdir(images_dir) if f.endswith('.nii')]

    # Group files
    group1 = [f for f in all_image_files if f.startswith("1") or f.startswith("2")]
    group2 = [f for f in all_image_files if f.startswith("3")]

    def split_and_copy(group_files, group_name):
        n_total = len(group_files)
        n_train = int(n_total * 0.9)
        random.shuffle(group_files)
        train_files = group_files[:n_train]
        test_files = group_files[n_train:]
        if args.verbose:
            print(f"[GRACE] {group_name}: {len(train_files)} train, {len(test_files)} test")

        for fname in train_files:
            shutil.copy(os.path.join(images_dir, fname), os.path.join(imagesTr_dir, fname))
            shutil.copy(os.path.join(labels_dir, fname), os.path.join(labelsTr_dir, fname))

        for fname in test_files:
            shutil.copy(os.path.join(images_dir, fname), os.path.join(imagesTs_dir, fname))
            shutil.copy(os.path.join(labels_dir, fname), os.path.join(labelsTs_dir, fname))

    # Process both groups
    split_and_copy(group1, "Group 1")
    split_and_copy(group2, "Group 2")
    if args.verbose:
        print("[GRACE] Data splitting completed.")

    # Step 3: Generate dataset.json
    if args.verbose:
        print("[GRACE] Generating dataset.json...")
    description = "AISEG V5 - Code Validation"
    license_text = "UF"
    modality = {"x0": "T1"}
    labels = {
        "x0": "background",
        "x1": "wm",
        "x2": "gm",
        "x3": "eyes",
        "x4": "csf",
        "x5": "air",
        "x6": "blood",
        "x7": "cancellous",
        "x8": "cortical",
        "x9": "skin",
        "x10": "fat",
        "x11": "muscle"
    }
    

    # Get test images
    test_files = sorted([f"./{img.relative_to(base_dir)}" for img in imagesTs_dir.glob('*.nii')])
    test = [str(f) for f in test_files]
    numTest = len(test_files)

    # Get training and label files
    train_images = sorted([f'./{img.relative_to(base_dir)}' for img in imagesTr_dir.glob("*.nii")])
    train_labels = sorted([f'./{lbl.relative_to(base_dir)}' for lbl in labelsTr_dir.glob("*.nii")])
    assert len(train_images) == len(train_labels), "Mismatch between imagesTr and labelsTr"

    # 90/10 split
    train_imgs, val_imgs, train_lbls, val_lbls = train_test_split(
        train_images, train_labels, test_size=0.10, random_state=42
    )

    # Build training and validation sets
    training = [{"image": str(img), "label": str(lbl)} for img, lbl in zip(train_imgs, train_lbls)]
    validation = [{"image": str(img), "label": str(lbl)} for img, lbl in zip(val_imgs, val_lbls)]
    numTraining = len(train_images)

    # Build full structure
    s = {
        "description": description,
        "license": license_text,
        "modality": modality,
        "labels": labels,
        "name": "ACT",
        "numTest": numTest,
        "numTraining": numTraining,
        "reference": "NA",
        "release": "NA",
        "tensorImageSize": "3D",
        "test": test,
        "training": training,
        "validation": validation
    }

    # Write to JSON
    with open(f"{base_dir}/dataset.json", "w") as f:
        json.dump(s, f, indent=4)
    if args.verbose:
        print("[GRACE] dataset.json created successfully.")

if __name__ == "__main__":
    main()
