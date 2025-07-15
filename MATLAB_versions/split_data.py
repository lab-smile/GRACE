import os
import shutil
import random

# Set seed for reproducibility
random.seed(42)

# Define base folders
base_dir = "GRACE_Data"
image_dir = os.path.join(base_dir, "images")
label_dir = os.path.join(base_dir, "labels")

# Output folders
dest_folders = {
    "imagesTr": os.path.join(base_dir, "imagesTr"),
    "imagesTs": os.path.join(base_dir, "imagesTs"),
    "labelsTr": os.path.join(base_dir, "labelsTr"),
    "labelsTs": os.path.join(base_dir, "labelsTs"),
}

# Make sure destination folders exist
for path in dest_folders.values():
    os.makedirs(path, exist_ok=True)

# List all image files
all_image_files = [f for f in os.listdir(image_dir) if f.endswith('.nii')]

# Group files
group1 = [f for f in all_image_files if f.startswith("1") or f.startswith("2")]
group2 = [f for f in all_image_files if f.startswith("3")]

def split_and_copy(group_files, group_name):
    n_total = len(group_files)
    n_train = int(n_total * 0.9)
    random.shuffle(group_files)
    train_files = group_files[:n_train]
    test_files = group_files[n_train:]

    print(f"{group_name}: {len(train_files)} train, {len(test_files)} test")

    for fname in train_files:
        shutil.copy(os.path.join(image_dir, fname), os.path.join(dest_folders["imagesTr"], fname))
        shutil.copy(os.path.join(label_dir, fname), os.path.join(dest_folders["labelsTr"], fname))

    for fname in test_files:
        shutil.copy(os.path.join(image_dir, fname), os.path.join(dest_folders["imagesTs"], fname))
        shutil.copy(os.path.join(label_dir, fname), os.path.join(dest_folders["labelsTs"], fname))

# Process both groups
split_and_copy(group1, "Group 1")
split_and_copy(group2, "Group 2")
