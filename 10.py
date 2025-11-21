import os 
import json
import numpy as np
import torch
import monai
from monai.data import DataLoader, Dataset
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, NormalizeIntensityd, ToTensord, Compose
)
from monai.networks.nets import DenseNet121
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from torch.optim.lr_scheduler import OneCycleLR   


with open("predict_age_data/eid2age.json", "r") as f:
    eid2age = json.load(f)

ages = np.array(list(eid2age.values()), dtype=np.float32)
age_mean, age_std = ages.mean(), ages.std()
print(f"年龄均值: {age_mean:.4f}, 年龄标准差: {age_std:.4f}")

def normalize_age(age):
    return (age - age_mean) / age_std

def denormalize_age(norm_age):
    return norm_age * age_std + age_mean


train_transforms = Compose([
    LoadImaged(keys="image"),
    EnsureChannelFirstd(keys=["image"]),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    ToTensord(keys=["image", "label"]),
])

val_transforms = Compose([
    LoadImaged(keys="image"),
    EnsureChannelFirstd(keys=["image"]),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    ToTensord(keys=["image", "label"]),
])

data_dir = "/data/weidong/UKB/data"
train_files = []
for eid, age in eid2age.items():
    img_path = os.path.join(data_dir, str(eid) + '_20252_2_0', "T1", "T1_brain_to_MNI.nii.gz")
    if os.path.exists(img_path):
        train_files.append({"image": img_path, "label": float(normalize_age(age))})

print(f"共找到 {len(train_files)} 个样本")


split_ratio = 0.8
split_idx = int(len(train_files) * split_ratio)
train_files_split = train_files[:split_idx]        
val_files_split = train_files[split_idx:]  

train_ds = Dataset(data=train_files_split, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=12, shuffle=True, num_workers=2)

val_ds = Dataset(data=val_files_split, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=12, shuffle=False, num_workers=2)


model = DenseNet121(
    spatial_dims=3,
    in_channels=1,
    out_channels=1
)
model = torch.nn.DataParallel(model)
model.to(device)


epochs = 50
min_lr = 1e-5
max_lr = 1e-3   # ✅ OneCycleLR 需要最大学习率
loss_fn = torch.nn.L1Loss()  
optimizer = torch.optim.Adam(model.parameters(), lr=max_lr)

scheduler = OneCycleLR(
    optimizer,
    max_lr=max_lr,
    epochs=epochs,
    steps_per_epoch=len(train_loader),
    pct_start=0.3,          # 前 30% 步骤升高学习率
    anneal_strategy='cos',  # 下降阶段使用余弦
    div_factor=25,          # 初始学习率 = max_lr / 25
    final_div_factor=1e4,   # 最小学习率 = max_lr / 1e4
)

val_interval = 2
best_metric = np.inf   
best_metric_epoch = -1
epoch_loss_values = []
val_mae_values = []
val_epochs = []
lr_history = []

writer = SummaryWriter(log_dir="runs/onecycle10")


for epoch in range(epochs):
    print(f"epoch {epoch+1}/{epochs}")
    model.train()
    epoch_loss = 0.0
    step = 0

    for batch_data in train_loader:
        step += 1
        inputs, label = batch_data["image"].to(device), batch_data["label"].to(device)
        label = label.unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, label.float())
        loss.backward()
        optimizer.step()
        scheduler.step()   

        epoch_loss += loss.item()
        print(f"step: {step}, train loss: {loss.item():.4f}")  
        global_step = epoch * len(train_loader) + step
        writer.add_scalar("Train/Loss_step", loss.item(), global_step)

    current_lr = optimizer.param_groups[0]['lr']
    lr_history.append(current_lr)
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)

    print(f"epoch {epoch+1} average loss: {epoch_loss:.4f}")
    writer.add_scalar("Train/Loss_epoch", epoch_loss, epoch+1)
    writer.add_scalar('lr', current_lr, epoch+1)


    if (epoch + 1) % val_interval == 0:
        model.eval()
        all_labels = []
        all_val_outputs = []

        for batch_data in val_loader:
            inputs, label = batch_data["image"].to(device), batch_data["label"].to(device)
            label = label.unsqueeze(1)
            all_labels.extend(denormalize_age(label.cpu().numpy().flatten()).tolist())
            with torch.no_grad():
                val_outputs = model(inputs)
                all_val_outputs.extend(denormalize_age(val_outputs.cpu().numpy().flatten()).tolist())

        mae = np.abs(np.subtract(all_labels, all_val_outputs)).mean()
        val_mae_values.append(mae)
        val_epochs.append(epoch+1)

        writer.add_scalar("Validation/MAE", mae, epoch+1)

        if mae < best_metric:
            best_metric = mae
            best_metric_epoch = epoch + 1
            torch.save(model.state_dict(), "best_metric_model_onecycle10.pth")
            print(" saved new best model")

        print(f"current epoch: {epoch+1}, current MAE: {mae:.4f}")
        print(f"best MAE: {best_metric:.4f} at epoch {best_metric_epoch}")

writer.close()  


# ====== 绘图部分 ======
plt.figure(figsize=(8,6))
plt.plot(range(1, len(epoch_loss_values)+1), epoch_loss_values, marker='o', label="Training Loss (MAE)")
plt.xlabel("Epoch")
plt.ylabel("Loss (normalized)")
plt.title("Training Loss Curve")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("final_training_loss_curve_onecycle10.png")
plt.close()

if len(val_mae_values) > 0:
    plt.figure(figsize=(8,6))
    plt.plot(val_epochs, val_mae_values, marker='s', color='orange', label="Validation MAE")
    plt.xlabel("Epoch")
    plt.ylabel("MAE (years)")
    plt.title("Validation MAE Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("final_validation_mae_curve_onecycle10.png")
    plt.close()

if len(lr_history) > 0:
    plt.figure(figsize=(8,6))
    plt.plot(range(1, len(lr_history)+1), lr_history, marker='.', color='green', label="Learning Rate (OneCycleLR)")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule (OneCycleLR)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("final_learning_rate_curve_onecycle10.png")
    plt.close()

print(f"Training completed, lowest mae: {best_metric:.4f} at epoch: {best_metric_epoch}")
