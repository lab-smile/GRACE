# coding=utf-8
"""
smoke_test.py — Environment and logic verification for GRACE.

Does NOT require real MRI data. Creates tiny synthetic NIfTI volumes in a
temporary directory, runs every major component of the training pipeline, then
cleans up.

Run with:
    python smoke_test.py

Expected runtime: < 2 min on CPU, < 30 s with a GPU.
Exit code 0 = all checks passed.
"""

import os
import shutil
import sys
import tempfile
import traceback

# Patch size for the smoke test. Must be divisible by 16 (UNETR ViT patch size).
# 32 is the minimum; small enough to run fast on CPU.
SPATIAL   = 32
N_CLASSES = 12
N_CASES   = 4    # synthetic NIfTI pairs to generate (3 train + 1 val)
STEPS     = 3    # mini training steps to execute

results = {}


def section(title: str):
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


def ok(msg: str):
    print(f"  [PASS] {msg}")


def fail(msg: str, exc: Exception | None = None) -> bool:
    print(f"  [FAIL] {msg}")
    if exc is not None:
        traceback.print_exc()
    return False


# ------------------------------------------------------------------
# 1. Imports
# ------------------------------------------------------------------
section("1. Package imports")
try:
    import math                         # noqa: F401
    import numpy as np
    import nibabel as nib
    import torch
    import torch.nn as nn               # noqa: F401
    import pandas as pd                 # noqa: F401
    from tqdm import tqdm               # noqa: F401
    ok("standard packages (numpy, nibabel, torch, pandas, tqdm)")

    import matplotlib
    matplotlib.use("Agg")               # headless — no display required
    import matplotlib.pyplot as plt     # noqa: F401
    ok("matplotlib (Agg backend)")

    from monai.config import print_config
    from monai.networks.nets import UNETR
    from monai.losses import DiceCELoss
    from monai.metrics import DiceMetric
    from monai.inferers import sliding_window_inference
    from monai.transforms import (
        AsDiscrete,
        Compose,
        CropForegroundd,
        EnsureChannelFirstd,
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
    from monai.data import (
        Dataset,
        DataLoader,
        decollate_batch,
        pad_list_data_collate,
    )
    ok("MONAI packages")

    results["imports"] = True

except Exception as e:
    fail("package import failed", e)
    results["imports"] = False
    print("\nCannot continue without required imports. Exiting.")
    sys.exit(1)

print()
print_config()

# ------------------------------------------------------------------
# 2. Device detection
# ------------------------------------------------------------------
section("2. Device detection")
try:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        ok(f"CUDA — {torch.cuda.get_device_name(0)}, "
           f"{torch.cuda.device_count()} GPU(s) available")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        ok("Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        ok("CPU (no GPU found — real training will be slow)")
    results["device"] = True
except Exception as e:
    fail("device detection", e)
    device = torch.device("cpu")
    results["device"] = False

# ------------------------------------------------------------------
# 3. Synthetic data + MONAI transform pipeline
# ------------------------------------------------------------------
section("3. MONAI transforms on synthetic NIfTI data")
tmp_dir = None
train_files = []
val_files   = []

try:
    tmp_dir = tempfile.mkdtemp(prefix="grace_smoke_")
    rng     = np.random.default_rng(seed=42)

    # Volumes must be larger than SPATIAL so RandCropByPosNegLabeld can sample
    vol_shape = (SPATIAL * 3, SPATIAL * 3, SPATIAL * 3)
    affine    = np.eye(4)

    data_list = []
    for i in range(N_CASES):
        img_arr  = rng.integers(0, 256,        vol_shape, dtype=np.uint8)
        # Ensure all class labels appear so foreground cropping always finds something
        lbl_arr  = rng.integers(0, N_CLASSES,  vol_shape, dtype=np.uint8)

        img_path = os.path.join(tmp_dir, f"img_{i:02d}.nii")
        lbl_path = os.path.join(tmp_dir, f"lbl_{i:02d}.nii")

        nib.save(nib.Nifti1Image(img_arr, affine), img_path)
        nib.save(nib.Nifti1Image(lbl_arr, affine), lbl_path)
        data_list.append({"image": img_path, "label": lbl_path})

    train_files = data_list[:-1]    # first N_CASES-1 for training
    val_files   = data_list[-1:]    # last one for validation

    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(SPATIAL, SPATIAL, SPATIAL),
            pos=1, neg=1, num_samples=1,
            image_key="image", image_threshold=0,
        ),
        RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.10),
        RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.10),
        RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.10),
        RandRotate90d(keys=["image", "label"], prob=0.10, max_k=3),
        RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.10),
        RandGaussianNoised(keys="image", prob=0.50, mean=0, std=0.1),
        ToTensord(keys=["image", "label"]),
    ])

    train_ds     = Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(
        train_ds, batch_size=2, shuffle=True, num_workers=0,
        collate_fn=pad_list_data_collate,
    )

    sample_batch = next(iter(train_loader))
    img_shape    = tuple(sample_batch["image"].shape)
    lbl_shape    = tuple(sample_batch["label"].shape)

    assert img_shape[-3:] == (SPATIAL, SPATIAL, SPATIAL), \
        f"Unexpected image shape: {img_shape}"
    ok(f"train transforms — image batch {img_shape}, label batch {lbl_shape}")

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        ToTensord(keys=["image", "label"]),
    ])

    val_ds     = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=pad_list_data_collate,
    )
    val_batch  = next(iter(val_loader))
    ok(f"val transforms   — image batch {tuple(val_batch['image'].shape)}")

    results["transforms"] = True

except Exception as e:
    fail("transforms", e)
    results["transforms"] = False

# ------------------------------------------------------------------
# 4. UNETR model instantiation
# ------------------------------------------------------------------
section("4. UNETR instantiation")
try:
    model = UNETR(
        in_channels=1,
        out_channels=N_CLASSES,
        img_size=(SPATIAL, SPATIAL, SPATIAL),
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        norm_name="instance",
        res_block=True,
        dropout_rate=0.0,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    ok(f"UNETR instantiated — {n_params / 1e6:.1f} M parameters")
    results["model_init"] = True

except Exception as e:
    fail("UNETR instantiation", e)
    results["model_init"] = False

# ------------------------------------------------------------------
# 5. Forward pass
# ------------------------------------------------------------------
section("5. UNETR forward pass")
try:
    dummy_x = torch.randn(1, 1, SPATIAL, SPATIAL, SPATIAL).to(device)

    model.eval()
    with torch.no_grad():
        out = model(dummy_x)

    expected = (1, N_CLASSES, SPATIAL, SPATIAL, SPATIAL)
    assert tuple(out.shape) == expected, f"Got {tuple(out.shape)}, expected {expected}"
    ok(f"output shape: {tuple(out.shape)}")
    results["forward"] = True

except Exception as e:
    fail("forward pass", e)
    results["forward"] = False

# ------------------------------------------------------------------
# 6. Loss + backward pass
# ------------------------------------------------------------------
section("6. DiceCELoss + backward pass")
try:
    loss_fn   = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    model.train()
    dummy_y = torch.randint(
        0, N_CLASSES, (1, 1, SPATIAL, SPATIAL, SPATIAL)
    ).float().to(device)

    logits = model(dummy_x)
    loss   = loss_fn(logits, dummy_y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    ok(f"loss = {loss.item():.4f}, backward + optimizer step OK")
    results["backward"] = True

except Exception as e:
    fail("loss / backward", e)
    results["backward"] = False

# ------------------------------------------------------------------
# 7. DiceMetric + post-processing
# ------------------------------------------------------------------
section("7. DiceMetric and post-processing (AsDiscrete)")
try:
    post_pred   = AsDiscrete(argmax=True, to_onehot=N_CLASSES)
    post_label  = AsDiscrete(to_onehot=N_CLASSES)
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

    model.eval()
    with torch.no_grad():
        pred = model(dummy_x)

    preds_list  = [post_pred(p)  for p in decollate_batch(pred)]
    labels_list = [post_label(l) for l in decollate_batch(dummy_y)]

    dice_metric(y_pred=preds_list, y=labels_list)
    score = dice_metric.aggregate().item()
    dice_metric.reset()

    ok(
        f"mean Dice on random data: {score:.4f} "
        f"(random baseline ≈ {1 / N_CLASSES:.4f})"
    )
    results["metrics"] = True

except Exception as e:
    fail("DiceMetric / post-processing", e)
    results["metrics"] = False

# ------------------------------------------------------------------
# 8. Sliding window inference
# ------------------------------------------------------------------
section("8. Sliding window inference")
try:
    model.eval()
    with torch.no_grad():
        sw_out = sliding_window_inference(
            dummy_x,
            (SPATIAL, SPATIAL, SPATIAL),
            sw_batch_size=1,
            predictor=model,
            overlap=0.25,
        )

    ok(f"sliding_window_inference output shape: {tuple(sw_out.shape)}")
    results["sliding_window"] = True

except Exception as e:
    fail("sliding window inference", e)
    results["sliding_window"] = False

# ------------------------------------------------------------------
# 9. Mini training loop on synthetic data loader
# ------------------------------------------------------------------
section(f"9. Mini training loop ({STEPS} steps on synthetic data loader)")
try:
    model.train()
    loader_iter = iter(train_loader)
    for step in range(STEPS):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)

        imgs   = batch["image"].to(device)
        lbls   = batch["label"].to(device)
        logits = model(imgs)
        loss   = loss_fn(logits, lbls)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        ok(f"step {step + 1}/{STEPS}  loss = {loss.item():.4f}")

    results["training_loop"] = True

except Exception as e:
    fail("mini training loop", e)
    results["training_loop"] = False

# ------------------------------------------------------------------
# Cleanup temp files
# ------------------------------------------------------------------
if tmp_dir and os.path.isdir(tmp_dir):
    shutil.rmtree(tmp_dir, ignore_errors=True)

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
section("Summary")
all_passed = True
for name, passed in results.items():
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}")
    if not passed:
        all_passed = False

if all_passed:
    print(
        "\n  All checks passed.\n"
        "  Your conda environment and GRACE model logic are ready to use."
    )
    sys.exit(0)
else:
    print(
        "\n  One or more checks FAILED — review the output above.\n"
        "  Install missing packages with:  pip install -r requirements.txt"
    )
    sys.exit(1)
