import os 
import logging
import sys
import numpy as np
import torch
import monai
from monai.data import DataLoader, Dataset
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, NormalizeIntensityd, RandFlipd, RandRotate90d,
    RandAffined, RandGaussianNoised, RandAdjustContrastd,
    ToTensord, Compose, SpatialPadd, RandSpatialCropd, CenterSpatialCropd
)
from monai.networks.nets import DenseNet121
import json

import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



with open("eid2age.json", "r") as f:
    eid2age = json.load(f)

ages = np.array(list(eid2age.values()), dtype=np.float32)
age_mean, age_std = ages.mean(), ages.std()
print(f"年龄均值: {age_mean:.4f}, 年龄标准差: {age_std:.4f}")



def normalize_age(age):
    return (age - age_mean) / age_std

def denormalize_age(norm_age):
    return norm_age * age_std + age_mean



data_dir = "/data/weidong/UKB/data"
train_files = []
for eid, age in eid2age.items():
    img_path = os.path.join(data_dir, str(eid) + '_20252_2_0', "T1", "T1_unbiased_brain.nii.gz")
    if os.path.exists(img_path):
        train_files.append({"image": img_path, "label": np.array([normalize_age(age)], dtype=np.float32)})


print(f"共找到 {len(train_files)} 个样本")

split_ratio = 0.8
split_idx = int(len(train_files) * split_ratio)
train_files_split = train_files[:split_idx]
val_files_split = train_files[split_idx:]


train_transforms = Compose([
    LoadImaged(keys="image"),
    EnsureChannelFirstd(keys=["image"]),
    Orientationd(keys=["image"], axcodes="RAS"),
    Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
    ScaleIntensityRanged(keys=["image"], a_min=0, a_max=4000, b_min=0.0, b_max=1.0, clip=True),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    SpatialPadd(keys=["image"], spatial_size=(128,128,128)),  
    RandSpatialCropd(keys=["image"], roi_size=(128,128,128), random_center=True, random_size=False),
    RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
    RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
    RandRotate90d(keys=["image"], prob=0.5, max_k=3),
    RandAffined(
        keys=["image"], prob=0.5,
        rotate_range=(0.1, 0.1, 0.1),
        scale_range=(0.1, 0.1, 0.1),
        mode="bilinear",
    ),
    RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.01),
    RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.5)),
    ToTensord(keys=["image", "label"]),
])

val_transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    Orientationd(keys=["image"], axcodes="RAS"),
    Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
    ScaleIntensityRanged(keys=["image"], a_min=0, a_max=4000, b_min=0.0, b_max=1.0, clip=True),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    SpatialPadd(keys=["image"], spatial_size=(128,128,128)),
    CenterSpatialCropd(keys=["image"], roi_size=(128,128,128)),
    ToTensord(keys=["image", "label"]),
])



train_ds = Dataset(data=train_files_split, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4)

val_ds = Dataset(data=val_files_split, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4)



model = DenseNet121(
    spatial_dims=3,
    in_channels=1,
    out_channels=1
).to(device)


loss_fn = torch.nn.L1Loss()  
optimizer = torch.optim.Adam(model.parameters(), 5e-5)



val_interval = 5
best_metric = np.inf   
best_metric_epoch = -1
epoch_loss_values = []
val_rmse_values = []
val_epochs = []
epochs = 50

for epoch in range(epochs):
    print(f"epoch {epoch+1}/{epochs}")
    model.train()
    epoch_loss = 0.0
    step = 0

    for batch_data in train_loader:
        step += 1
        inputs, label = batch_data["image"].to(device), batch_data["label"].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, label.float())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print(f"step: {step}, train loss: {loss.item():.4f}")

    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch+1} average loss: {epoch_loss:.4f}")


    if (epoch + 1) % val_interval == 0:
        model.eval()
        all_labels = []
        all_val_outputs = []

        for batch_data in val_loader:
            inputs, label = batch_data["image"].to(device), batch_data["label"].to(device)
            all_labels.extend(denormalize_age(label.cpu().detach().numpy().flatten()).tolist())
            with torch.no_grad():
                val_outputs = model(inputs)
                all_val_outputs.extend(denormalize_age(val_outputs.cpu().detach().numpy().flatten()).tolist())

        mse = np.square(np.subtract(all_labels, all_val_outputs)).mean()
        rmse = np.sqrt(mse)
        val_rmse_values.append(rmse)
        val_epochs.append(epoch+1)

        if rmse < best_metric:
            best_metric = rmse
            best_metric_epoch = epoch + 1
            torch.save(model.state_dict(), "best_metric_model.pth")
            print(" saved new best model")

        print(f"current epoch: {epoch+1}, current RMSE: {rmse:.4f}")
        print(f"best RMSE: {best_metric:.4f} at epoch {best_metric_epoch}")

print(f"Training completed, lowest RMSE: {best_metric:.4f} at epoch: {best_metric_epoch}")




# ------------------- 绘制训练 Loss 曲线 -------------------
plt.figure(figsize=(8,6))
plt.plot(range(1, len(epoch_loss_values)+1), epoch_loss_values, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Training Loss (MAE, normalized)")
plt.title("Training Loss Curve")
plt.grid(True)
plt.tight_layout()
plt.savefig("training_loss_curve.png")   # 保存图片
plt.show()


# ------------------- 绘制验证 RMSE 曲线 -------------------
if len(val_rmse_values) > 0:
    plt.figure(figsize=(8,6))
    plt.plot(val_epochs, val_rmse_values, marker='s', color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("Validation RMSE (years)")
    plt.title("Validation RMSE Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("validation_rmse_curve.png")  # 保存图片
    plt.show()
