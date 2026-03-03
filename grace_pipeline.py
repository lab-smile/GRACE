# coding=utf-8
"""
grace_pipeline.py — Standalone GRACE training (+ optional inference) script.

Edit only the CONFIG block below, then run:
    python grace_pipeline.py

Set RUN_TEST = True to also run inference on the held-out test split immediately
after training completes, using the best saved model weights.

Data directory check on startup:
  • dataset.json present              → used as-is.
  • dataset.json missing, but imagesTr/ labelsTr/ imagesTs/ labelsTs/ all contain
    matching .nii files               → dataset.json is auto-generated (90/10 train/val,
                                        random_state=42) and training proceeds.
  • Required directories missing or empty → clear error, no training starts.

Training outputs written to DATA_DIR:
    {MODEL_SAVE_NAME}.pth                    — best model checkpoint
    {MODEL_SAVE_NAME}_Loss.csv               — training loss per eval interval
    {MODEL_SAVE_NAME}_ValidationDice.csv     — validation Dice per eval interval
    {MODEL_SAVE_NAME}_training_metrics.pdf   — loss + Dice plots

Inference outputs (when RUN_TEST = True):
    TestResults/{MODEL_SAVE_NAME}/<case>.nii.gz  — per-case segmentation maps
"""

# ============================================================
# CONFIG — only edit this block before running
# ============================================================

DATA_DIR        = "/path/to/your/preprocessed/data/"
# Must point to a folder that contains either:
#   (a) dataset.json + imagesTr/ labelsTr/ imagesTs/ labelsTs/, or
#   (b) imagesTr/ labelsTr/ imagesTs/ labelsTs/ (dataset.json will be created).
# Run preprocess.py first if those sub-directories don't exist yet.

MODEL_SAVE_NAME = "grace"      # Output filename stem (no extension)
NUM_GPU         = 1            # GPUs to use (DataParallel); ignored on CPU/MPS
MAX_ITERATIONS  = 25000        # Total training steps
BATCH_TRAIN     = 10           # Training batch size (reduce if OOM)
BATCH_VAL       = 5            # Validation batch size (reduce if OOM)
SPATIAL_SIZE    = 64           # Patch size in voxels (must be divisible by 16)
N_CLASSES       = 12           # Background + 11 tissue classes
A_MIN           = 0            # Raw image intensity minimum (UINT8 data → 0)
A_MAX           = 255          # Raw image intensity maximum (UINT8 data → 255)
JSON_NAME       = "dataset.json"
RUN_TEST        = False        # Set True to run inference on the test split after training
                               # Outputs → DATA_DIR/TestResults/{MODEL_SAVE_NAME}/<case>.nii.gz

# ============================================================
# (Nothing below needs to be changed.)
# ============================================================

import json
import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless-safe; remove this line to get interactive plots
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from monai.config import print_config
from monai.data import (
    DataLoader,
    Dataset,
    decollate_batch,
    load_decathlon_datalist,
    pad_list_data_collate,
)
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNETR
from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandRotate90d,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    ToTensord,
)

# ------------------------------------------------------------------
# Data directory / JSON validation and auto-generation
# ------------------------------------------------------------------

REQUIRED_DIRS = ("imagesTr", "labelsTr", "imagesTs", "labelsTs")


def _generate_dataset_json(data_path: Path, json_path: Path) -> None:
    """Build dataset.json from the four nnU-Net sub-directories and write it."""
    images_tr = sorted((data_path / "imagesTr").glob("*.nii"))
    labels_tr = sorted((data_path / "labelsTr").glob("*.nii"))
    images_ts = sorted((data_path / "imagesTs").glob("*.nii"))

    if len(images_tr) != len(labels_tr):
        raise ValueError(
            f"imagesTr ({len(images_tr)} files) and labelsTr ({len(labels_tr)} files) "
            "have different counts. Cannot generate dataset.json."
        )

    # Relative paths (same convention as preprocess.py)
    tr_img_rel  = [f"./{p.relative_to(data_path)}" for p in images_tr]
    tr_lbl_rel  = [f"./{p.relative_to(data_path)}" for p in labels_tr]
    ts_img_rel  = [f"./{p.relative_to(data_path)}" for p in images_ts]

    # 90 / 10 train-validation split (reproducible)
    train_imgs, val_imgs, train_lbls, val_lbls = train_test_split(
        tr_img_rel, tr_lbl_rel, test_size=0.10, random_state=42
    )

    dataset = {
        "description": "GRACE — auto-generated dataset split",
        "license": "see MODEL_LICENSE",
        "modality": {"0": "T1"},
        "labels": {
            "0": "background", "1": "wm",   "2": "gm",     "3": "eyes",
            "4": "csf",        "5": "air",  "6": "blood",  "7": "cancellous",
            "8": "cortical",   "9": "skin", "10": "fat",   "11": "muscle",
        },
        "name": "GRACE",
        "numTest":     len(images_ts),
        "numTraining": len(images_tr),
        "reference": "https://github.com/lab-smile/GRACE",
        "release": "auto",
        "tensorImageSize": "3D",
        "test":       ts_img_rel,
        "training":   [{"image": i, "label": l} for i, l in zip(train_imgs, train_lbls)],
        "validation": [{"image": i, "label": l} for i, l in zip(val_imgs,   val_lbls)],
    }

    with open(json_path, "w") as f:
        json.dump(dataset, f, indent=4)

    print(
        f"[grace_pipeline] dataset.json created: "
        f"{len(train_imgs)} train, {len(val_imgs)} val, {len(images_ts)} test."
    )


def _check_data_dir(data_path: Path, json_path: Path) -> None:
    """Validate DATA_DIR and auto-generate dataset.json when possible."""
    if not data_path.is_dir():
        print(f"[ERROR] DATA_DIR does not exist: {data_path}", file=sys.stderr)
        sys.exit(1)

    if json_path.is_file():
        # Fast path: JSON already present
        print(f"[grace_pipeline] Found existing {json_path.name} — skipping JSON generation.")
        return

    # JSON is missing — check whether the required sub-directories are populated
    missing_dirs  = [d for d in REQUIRED_DIRS if not (data_path / d).is_dir()]
    empty_dirs    = [d for d in REQUIRED_DIRS if (data_path / d).is_dir()
                     and not any((data_path / d).glob("*.nii"))]

    if missing_dirs:
        print(
            f"[ERROR] {JSON_NAME} not found in {data_path} and the following required "
            f"sub-directories are also missing: {missing_dirs}\n"
            "Run preprocess.py first, or manually create imagesTr/, labelsTr/, "
            "imagesTs/, labelsTs/ with your .nii files.",
            file=sys.stderr,
        )
        sys.exit(1)

    if empty_dirs:
        print(
            f"[ERROR] {JSON_NAME} not found and these directories contain no .nii files: "
            f"{empty_dirs}\n"
            "Populate them with your data before running training.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[grace_pipeline] {JSON_NAME} not found — auto-generating from sub-directories …")
    _generate_dataset_json(data_path, json_path)


# ------------------------------------------------------------------
# Resolve paths + validate
# ------------------------------------------------------------------

data_path = Path(DATA_DIR)
json_path = data_path / JSON_NAME

_check_data_dir(data_path, json_path)

# ------------------------------------------------------------------
# Device
# ------------------------------------------------------------------

def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = pick_device()
print(f"[grace_pipeline] Using device: {device}")

start_time = time.time()
print_config()

# ------------------------------------------------------------------
# Transforms
# ------------------------------------------------------------------

train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=A_MIN,
            a_max=A_MAX,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(SPATIAL_SIZE, SPATIAL_SIZE, SPATIAL_SIZE),
            pos=1,
            neg=1,
            num_samples=1,
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.10),
        RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.10),
        RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.10),
        RandRotate90d(keys=["image", "label"], prob=0.10, max_k=3),
        RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.10),
        RandGaussianNoised(keys="image", prob=0.50, mean=0, std=0.1),
        ToTensord(keys=["image", "label"]),
    ]
)

val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=A_MIN,
            a_max=A_MAX,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        ToTensord(keys=["image", "label"]),
    ]
)

# ------------------------------------------------------------------
# Data loaders
# ------------------------------------------------------------------

train_files = load_decathlon_datalist(str(json_path), True, "training")
val_files   = load_decathlon_datalist(str(json_path), True, "validation")

print(f"[grace_pipeline] Training cases:   {len(train_files)}")
print(f"[grace_pipeline] Validation cases: {len(val_files)}")

train_ds = Dataset(data=train_files, transform=train_transforms)
val_ds   = Dataset(data=val_files,   transform=val_transforms)

pin = device.type in ("cuda", "mps")

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_TRAIN,
    shuffle=True,
    num_workers=0,
    pin_memory=pin,
    collate_fn=pad_list_data_collate,
)
val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_VAL,
    shuffle=False,
    num_workers=0,
    pin_memory=pin,
    collate_fn=pad_list_data_collate,
)

# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------

base_model = UNETR(
    in_channels=1,
    out_channels=N_CLASSES,
    img_size=(SPATIAL_SIZE, SPATIAL_SIZE, SPATIAL_SIZE),
    feature_size=16,
    hidden_size=768,
    mlp_dim=3072,
    num_heads=12,
    norm_name="instance",
    res_block=True,
    dropout_rate=0.0,
)

if device.type == "cuda" and NUM_GPU > 1:
    model = nn.DataParallel(base_model, device_ids=list(range(NUM_GPU)))
    model = model.to(device)
else:
    model = base_model.to(device)

loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
optimizer     = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

# ------------------------------------------------------------------
# Post-processing and metrics
# ------------------------------------------------------------------

post_label  = AsDiscrete(to_onehot=N_CLASSES)
post_pred   = AsDiscrete(argmax=True, to_onehot=N_CLASSES)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

max_iterations = MAX_ITERATIONS
eval_num       = math.ceil(MAX_ITERATIONS * 0.02)   # validate every 2 % of training

# ------------------------------------------------------------------
# Validation loop
# ------------------------------------------------------------------

def validation(epoch_iterator_val, global_step):
    model.eval()
    dice_vals = []
    with torch.no_grad():
        for batch in epoch_iterator_val:
            val_inputs  = batch["image"].to(device)
            val_labels  = batch["label"].to(device)
            sw_bs       = 4 if device.type == "cuda" else 1
            val_outputs = sliding_window_inference(
                val_inputs,
                (SPATIAL_SIZE, SPATIAL_SIZE, SPATIAL_SIZE),
                sw_bs,
                model,
            )
            labels_list  = [post_label(l) for l in decollate_batch(val_labels)]
            outputs_list = [post_pred(p)  for p in decollate_batch(val_outputs)]
            dice_metric(y_pred=outputs_list, y=labels_list)
            dice = dice_metric.aggregate().item()
            dice_vals.append(dice)
            epoch_iterator_val.set_description(
                f"Validate ({global_step} / {max_iterations} Steps) (dice={dice:.5f})"
            )
        dice_metric.reset()
    return float(np.mean(dice_vals))

# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------

def train(global_step, train_loader, dice_val_best, global_step_best):
    model.train()
    epoch_loss = 0.0
    step       = 0
    epoch_iterator = tqdm(
        train_loader,
        desc="Training (X / X Steps) (loss=X.X)",
        dynamic_ncols=True,
    )
    for step, batch in enumerate(epoch_iterator):
        step += 1
        x = batch["image"].to(device)
        y = batch["label"].to(device)
        logits = model(x)
        loss   = loss_function(logits, y)
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        epoch_iterator.set_description(
            f"Training ({global_step} / {max_iterations} Steps) (loss={loss.detach().item():.5f})"
        )
        if (global_step % eval_num == 0 and global_step != 0) or global_step == max_iterations:
            epoch_iterator_val = tqdm(
                val_loader,
                desc="Validate (X / X Steps) (dice=X.X)",
                dynamic_ncols=True,
            )
            dice_val    = validation(epoch_iterator_val, global_step)
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)
            if dice_val > dice_val_best:
                dice_val_best    = dice_val
                global_step_best = global_step
                torch.save(
                    model.state_dict(),
                    data_path / f"{MODEL_SAVE_NAME}.pth",
                )
                print(
                    f"Model saved — best Dice: {dice_val_best:.5f}  "
                    f"current Dice: {dice_val:.5f}"
                )
            else:
                print(
                    f"Model not saved — best Dice: {dice_val_best:.5f}  "
                    f"current Dice: {dice_val:.5f}"
                )
        global_step += 1
    return global_step, dice_val_best, global_step_best

# ------------------------------------------------------------------
# Run training
# ------------------------------------------------------------------

global_step       = 0
dice_val_best     = 0.0
global_step_best  = 0
epoch_loss_values = []
metric_values     = []

while global_step < max_iterations:
    global_step, dice_val_best, global_step_best = train(
        global_step, train_loader, dice_val_best, global_step_best
    )

model.load_state_dict(
    torch.load(data_path / f"{MODEL_SAVE_NAME}.pth", map_location=device)
)

# ------------------------------------------------------------------
# Save plots + CSVs
# ------------------------------------------------------------------

x_axis = [eval_num * (i + 1) for i in range(len(epoch_loss_values))]

plt.figure("train", (12, 6))

plt.subplot(1, 2, 1)
plt.title("Iteration Average Loss")
plt.xlabel("Iteration")
plt.plot(x_axis, epoch_loss_values)

plt.subplot(1, 2, 2)
plt.title("Val Mean Dice")
plt.xlabel("Iteration")
plt.plot(x_axis, metric_values)

plt.savefig(data_path / f"{MODEL_SAVE_NAME}_training_metrics.pdf")
plt.close()

pd.DataFrame({"Iteration": x_axis, "Loss": epoch_loss_values}).to_csv(
    data_path / f"{MODEL_SAVE_NAME}_Loss.csv", index=False
)
pd.DataFrame({"Iteration": x_axis, "Dice": metric_values}).to_csv(
    data_path / f"{MODEL_SAVE_NAME}_ValidationDice.csv", index=False
)

# ------------------------------------------------------------------
# Inference on test split (runs only when RUN_TEST = True)
# ------------------------------------------------------------------

def run_inference() -> None:
    """Run sliding-window inference on every case in the 'test' split of dataset.json.

    Mirrors test.py exactly:
      - Image-only transforms (no label, no augmentation)
      - Sliding window with 80 % overlap
      - Output preserves the original NIfTI header and affine
      - Saves compressed NIfTI to DATA_DIR/TestResults/{MODEL_SAVE_NAME}/
    """
    print("\n[grace_pipeline] Starting test-set inference …")

    # Test transforms — image only, no augmentation
    test_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
            Orientationd(keys=["image"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"], a_min=A_MIN, a_max=A_MAX,
                b_min=0.0, b_max=1.0, clip=True,
            ),
            EnsureTyped(keys=["image"]),
        ]
    )

    test_files = load_decathlon_datalist(str(json_path), True, "test")
    if not test_files:
        print("[grace_pipeline] No test cases found in dataset.json — skipping inference.")
        return

    print(f"[grace_pipeline] Test cases: {len(test_files)}")

    test_ds  = Dataset(data=test_files, transform=test_transforms)
    save_dir = data_path / "TestResults" / MODEL_SAVE_NAME
    save_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    sw_bs = 4 if device.type == "cuda" else 1

    for i in range(len(test_ds)):
        with torch.no_grad():
            img      = test_ds[i]["image"]
            img_path = Path(img.meta["filename_or_obj"])
            inputs   = torch.unsqueeze(img, 0).to(device)   # (1, C, H, W, D)
            outputs  = sliding_window_inference(
                inputs,
                (SPATIAL_SIZE, SPATIAL_SIZE, SPATIAL_SIZE),
                sw_bs,
                model,
                overlap=0.8,
            )

        seg = torch.argmax(outputs, dim=1).detach().cpu().numpy()[0]

        # Preserve original header + affine so the output is spatially registered
        ref      = nib.load(str(img_path))
        new_nii  = nib.Nifti1Image(seg.astype(np.uint8), ref.affine, ref.header.copy())

        # Strip .nii (or .nii.gz) and save as compressed NIfTI
        stem     = img_path.stem if img_path.suffix != ".gz" else Path(img_path.stem).stem
        out_path = save_dir / f"{stem}.nii.gz"
        nib.save(new_nii, str(out_path))
        print(f"[grace_pipeline] Saved {i + 1}/{len(test_ds)}: {out_path.name}")

    print(f"[grace_pipeline] Inference complete. Results in: {save_dir}")


elapsed = time.time() - start_time
print(f"\nDone. Best Dice: {dice_val_best:.5f} at step {global_step_best}.")
print(f"Total time: {elapsed/3600:.2f} h ({elapsed:.0f} s)")

if RUN_TEST:
    run_inference()
