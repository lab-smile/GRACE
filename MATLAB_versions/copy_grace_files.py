import os
import shutil

# Define base directory
base_dir = "GRACE_Data"

# Source parent folders to search
source_folders = ["PL_ETnew", "PL_ETold", "PL_v1", "PL_v2", "PL_v3"]

# Destination folders
images_dir = os.path.join(base_dir, "images")
labels_dir = os.path.join(base_dir, "labels")

# Create destination folders if they don't exist
os.makedirs(images_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

# Loop through each source folder
for folder in source_folders:
    full_folder_path = os.path.join(base_dir, folder)
    if not os.path.isdir(full_folder_path):
        continue
    
    for subfolder in os.listdir(full_folder_path):
        subfolder_path = os.path.join(full_folder_path, subfolder)

        # Skip folders with '_MISSING' in the name
        if '_MISSING' in subfolder or not os.path.isdir(subfolder_path):
            continue

        try:
            # Extract numeric ID
            numeric_id = subfolder.split('_')[1].replace('sub-', '')

            # Define source file paths
            t1_path = os.path.join(subfolder_path, "T1.nii")
            mask_path = os.path.join(subfolder_path, "T1_T1orT2_masks.nii")

            # Define destination paths
            new_t1_path = os.path.join(images_dir, f"{numeric_id}.nii")
            new_mask_path = os.path.join(labels_dir, f"{numeric_id}.nii")

            # Copy files
            if os.path.exists(t1_path):
                shutil.copy(t1_path, new_t1_path)
                print(f"Copied {t1_path} -> {new_t1_path}")
            else:
                print(f"Missing T1 file: {t1_path}")

            if os.path.exists(mask_path):
                shutil.copy(mask_path, new_mask_path)
                print(f"Copied {mask_path} -> {new_mask_path}")
            else:
                print(f"Missing mask file: {mask_path}")

        except Exception as e:
            print(f"Error processing {subfolder_path}: {e}")
