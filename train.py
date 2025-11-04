# coding=utf-8

#load packages:

#standard packages - 

import os
import math
import time
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

#load monai functions - 

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    Spacingd,
    RandFlipd,
    ToTensord,
    AsDiscrete,
    LoadImaged,
    Orientationd,
    RandRotate90d,
    CropForegroundd,
    RandGaussianNoised,
    EnsureChannelFirstd,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    RandCropByPosNegLabeld
)

from monai.metrics import DiceMetric
from monai.config import print_config
from monai.networks.nets import UNETR
from monai.data import (
    Dataset,
    DataLoader,
    decollate_batch,
    load_decathlon_datalist,
    pad_list_data_collate
)
#-----------------------------------

def pick_device():
    # auto: try CUDA, then MPS (Just an example, you may change this per your preference), then CPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = pick_device()
print(f"DEBUG Using device: {device}")

#set up starting conditions:

start_time = time.time()
print_config()

# our CLI parser
parser = argparse.ArgumentParser()
parser.add_argument("--num_gpu", type=int, default=3, help="number of gpus")
parser.add_argument("--spatial_size", type=int, default=64, help="one patch dimension")
parser.add_argument("--a_min_value", type=int, default=0, help="minimum image intensity")
parser.add_argument("--N_classes", type=int, default=12, help="number of tissues classes")
parser.add_argument("--a_max_value", type=int, default=255, help="maximum image intensity")
parser.add_argument("--max_iteration", type=int, default=25000, help="number of iterations")
parser.add_argument("--batch_size_train", type=int, default=10, help="batch size training data")
parser.add_argument("--model_save_name", type=str, default="unetr_v5_cos", help="model save name")
parser.add_argument("--batch_size_validation", type=int, default=5, help="batch size validation data")
parser.add_argument("--json_name", type=str, default="dataset.json", help="name of the file used to map data splits")
parser.add_argument("--data_dir", type=str, default="/red/nvidia-ai/SkylarStolte/training_pairs_v5/", help="directory the dataset is in")
args = parser.parse_args()

split_JSON = args.json_name #"dataset.json". Make sure that the JSON file, with exact name, is in the data_dir folder
datasets = args.data_dir + split_JSON # Add / to data_dir if not present or change this line to hardcode the path
num_classes = args.N_classes

#-----------------------------------

#data transformations:

train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS", labels=None),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=args.a_min_value,
            a_max=args.a_max_value, #my original data is in UINT8
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"), #can crop data since taking patches that are less than full
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(args.spatial_size, args.spatial_size, args.spatial_size),
            pos=1,
            neg=1,
            # reduce number of samples to lower memory use (was 16)
            num_samples=1, # much smaller -> minimal memory (batch_size x num_samples x patch)
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[0],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[1],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[2],
            prob=0.10,
        ),
        RandRotate90d(
            keys=["image", "label"],
            prob=0.10,
            max_k=3,
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=0.10,
        ),
        RandGaussianNoised(keys = "image", prob = .50, mean = 0, std = 0.1),
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
            keys=["image"], a_min=args.a_min_value, a_max=args.a_max_value, b_min=0.0, b_max=1.0, clip=True
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        ToTensord(keys=["image", "label"]),
    ]
)

#-----------------------------------

#set up data loaders

train_files = load_decathlon_datalist(datasets, True, "training")
val_files = load_decathlon_datalist(datasets, True, "validation")
train_ds = Dataset(
    data=train_files,
    transform=train_transforms,
)
train_loader = DataLoader(
    train_ds,
    batch_size=args.batch_size_train,
    shuffle=True,
    # use single-process loader inside small containers (lower memory). Increase if you have enough RAM.
    num_workers=0,
    # only pin memory when an accelerator is available
    pin_memory=(device.type == "cuda" or (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())),
    collate_fn=pad_list_data_collate,
)
val_ds = Dataset(
    data=val_files, transform=val_transforms, 
)
val_loader = DataLoader(
    val_ds,
    batch_size=args.batch_size_validation,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda" or (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())),
    collate_fn=pad_list_data_collate,
)

#-----------------------------------

#set up gpu device and unetr model

# build base model
base_model = UNETR(
    in_channels=1,
    out_channels=args.N_classes, #12 for all tissues
    img_size=(args.spatial_size, args.spatial_size, args.spatial_size),
    feature_size=16,
    hidden_size=768,
    mlp_dim=3072,
    num_heads=12,
    norm_name="instance",
    res_block=True,
    dropout_rate=0.0,
)

# Wrap with DataParallel only when CUDA is available and multiple GPUs requested.
if device.type == "cuda" and args.num_gpu > 1 and torch.cuda.is_available():
    model = nn.DataParallel(base_model, device_ids=[i for i in range(args.num_gpu)])
    model = model.to(device)
else:
    # keep plain model for CPU or single-GPU runs
    model = base_model.to(device)

loss_function = DiceCELoss(to_onehot_y=num_classes, softmax=True) #Focal #DiceCELoss(to_onehot_y=True, softmax=True)
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

#-----------------------------------

def validation(epoch_iterator_val):
    model.eval()
    dice_vals = list()
    with torch.no_grad():
        for _, batch in enumerate(epoch_iterator_val):
            val_inputs, val_labels = (batch["image"].to(device), batch["label"].to(device))
            # choose a small sliding-window batch size on CPU to avoid OOM
            sw_batch_size = 4 if device.type == "cuda" else 1
            val_outputs = sliding_window_inference(val_inputs, (args.spatial_size, args.spatial_size, args.spatial_size), sw_batch_size, model)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [
                post_label(val_label_tensor) for val_label_tensor in val_labels_list
            ]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [
                post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
            ]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            dice = dice_metric.aggregate().item()
            dice_vals.append(dice)
            epoch_iterator_val.set_description(
                "Validate (%d / %d Steps) (dice=%2.5f)" % (global_step, 10.0, dice)
            )
        dice_metric.reset()
    mean_dice_val = np.mean(dice_vals)
    return mean_dice_val

#-----------------------------------

def train(global_step, train_loader, dice_val_best, global_step_best):
    model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, y = (batch["image"].to(device), batch["label"].to(device))
        logit_map = model(x)
        loss = loss_function(logit_map, y)
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        loss_val = loss.detach().item()
        epoch_iterator.set_description(
            "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, loss_val)
        )
        if (
            global_step % eval_num == 0 and global_step != 0
        ) or global_step == max_iterations:
            epoch_iterator_val = tqdm(
                val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
            )
            dice_val = validation(epoch_iterator_val)
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_step
                torch.save(
                    model.state_dict(), os.path.join(args.data_dir, args.model_save_name + ".pth")
                )
                print(
                    "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
            else:
                print(
                    "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
        global_step += 1
    return global_step, dice_val_best, global_step_best

#-----------------------------------

max_iterations = args.max_iteration #25000
eval_num = math.ceil(args.max_iteration * 0.02)#500
post_label = AsDiscrete(to_onehot=num_classes, num_classes=args.N_classes) 
post_pred = AsDiscrete(argmax=True, to_onehot=num_classes, num_classes=args.N_classes) 
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
global_step = 0
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []
while global_step < max_iterations:
    global_step, dice_val_best, global_step_best = train(
        global_step, train_loader, dice_val_best, global_step_best
    )
# load checkpoint using map_location so GPU-saved checkpoints can be loaded on CPU
model.load_state_dict(torch.load(os.path.join(args.data_dir, args.model_save_name + ".pth"), map_location=device))


#-----------------------------------

#training loss and validation evaluation

plt.figure("train", (12, 6))
plt.subplot(1, 2, 1)
plt.title("Iteration Average Loss")
x = [eval_num * (i + 1) for i in range(len(epoch_loss_values))]
y = epoch_loss_values
plt.xlabel("Iteration")
plt.plot(x, y)

dict = {'Iteration': x, 'Loss': y}  
df = pd.DataFrame(dict) 
df.to_csv(os.path.join(args.data_dir,args.model_save_name + '_Loss.csv'))

plt.subplot(1, 2, 2)
plt.title("Val Mean Dice")
x = [eval_num * (i + 1) for i in range(len(metric_values))]
y = metric_values
plt.xlabel("Iteration")
plt.plot(x, y)
#plt.show() #uncomment to see the plot immediately
plt.savefig(os.path.join(args.data_dir, args.model_save_name + "_training_metrics.pdf"))

dict = {'Iteration': x, 'Dice': y}  
df = pd.DataFrame(dict) 
df.to_csv(os.path.join(args.data_dir,args.model_save_name + '_ValidationDice.csv'))

#------------------------------------
#time since start
print("--- %s seconds ---" % (time.time() - start_time))
