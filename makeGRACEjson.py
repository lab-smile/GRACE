import os
import json
import glob
from pathlib import Path
from sklearn.model_selection import train_test_split

# Define paths
imagesTr_dir = Path('imagesTr')
labelsTr_dir = Path('labelsTr')
imagesTs_dir = Path('imagesTs')

# Define metadata
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
test_files = sorted(imagesTs_dir.glob("*.nii"))
test = [str(f) for f in test_files]
numTest = len(test_files)

# Get training and label files
train_images = sorted(imagesTr_dir.glob("*.nii"))
train_labels = sorted(labelsTr_dir.glob("*.nii"))
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
with open("dataset.json", "w") as f:
    json.dump(s, f, indent=4)

print("✅ dataset.json created successfully.")
