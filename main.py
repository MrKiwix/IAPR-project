import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path
from skimage import io, transform
from src.helper import display_sample
from torch import nn


class ChocolateDataset(Dataset):

    def __init__(self, data_dir, label_csv, transform=None, target_transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.label_df = pd.read_csv(label_csv)
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.label_df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist
        
        img_path = Path(f"{self.data_dir}/L{self.label_df.iloc[idx, 0]}.JPG")
        
        image = io.imread(img_path)
        label = self.label_df.iloc[idx, 1:]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:   
            label = self.target_transform(label)

        return image, label
    
class LabelToTensor:
    def __call__(self, label):
        return torch.tensor(label.to_numpy())
    
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)

        # first conv layer, downsampling if stride > 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x += identity
        x = self.relu(x)

        return x

import torch.nn as nn
import torch

class CountHead(nn.Module):
    """
    in_channels : #channels coming from the encoder
    hidden      : size of the intermediate layer (default 512)
    n_classes   : how many categories we count
    """
    def __init__(self, in_channels=512, hidden=512, n_classes=3, p_drop=0.2):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)           # (B, C, H, W) → (B, C, 1, 1)

        self.regressor = nn.Sequential(              # (B, C) → (B, n_classes)
            nn.Flatten(1),                           # (B, C, 1, 1) → (B, C)
            nn.Linear(in_channels, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(hidden, n_classes, bias=True)  # final counts (float)
        )

    def forward(self, x):
        x = self.gap(x)
        return self.regressor(x)                     # shape (B, n_classes)


class ChocoNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)

        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(ResBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(ResBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(ResBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(ResBlock, 512, 2, stride=2)

        self.head = CountHead(in_channels=512, n_classes=13)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        
        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.head(x)

        return x
    
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")



# -----------------  PREPARE THE DATA  -----------------
from torch.utils.data import random_split

NUM_CLASSES = 13
IMG_SIZE    = (120, 180)          # height, width  (change as you like)

label_csv  = Path("./data/train.csv")
images_dir = Path("./data/train")

# transforms: uint8 [0-255] -> float32 [0-1]  + simple resize
img_tf = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize(IMG_SIZE, antialias=True),
])

dataset = ChocolateDataset(
    data_dir=images_dir,
    label_csv=label_csv,
    transform=img_tf,
    target_transform=LabelToTensor()
)

# split 80 % / 20 %
train_len = int(0.8 * len(dataset))
test_len  = len(dataset) - train_len
train_ds, test_ds = random_split(dataset, [train_len, test_len],
                                 generator=torch.Generator().manual_seed(42))

batch_size = 32           # ↑ from 4  (better gradient statistics)
num_workers = 4           # set 0 on Windows if you hit issues

train_loader = DataLoader(train_ds, batch_size,
                          shuffle=True,  num_workers=num_workers, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size,
                          shuffle=False, num_workers=num_workers, pin_memory=True)

# -----------------  BUILD MODEL  -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = ChocoNetwork().to(device)

# -----------------  OPTIMISER & SCHEDULER  -----------------
# freeze backbone for 5 epochs
for name, p in model.named_parameters():
    if not name.startswith("head."):
        p.requires_grad_(False)

# two parameter groups: head LR 1e-3, backbone LR 1e-4 (when unfrozen)
optim_groups = [
    {"params": [p for n, p in model.named_parameters() if n.startswith("head.")],
     "lr": 1e-3},
    {"params": [p for n, p in model.named_parameters() if not n.startswith("head.")],
     "lr": 1e-4},
]
optimizer = torch.optim.AdamW(optim_groups, weight_decay=1e-4)

# Smooth L1 (Huber) with β=1.0
criterion = nn.SmoothL1Loss(beta=1.0)

# -----------------  TRAIN / EVAL LOOPS  -----------------
def train_epoch(loader, net, loss_fn, optim, epoch):
    net.train()
    running_loss = 0.0
    for imgs, targets in loader:
        imgs     = imgs.to(device, non_blocking=True)
        targets  = targets.float().to(device, non_blocking=True)

        preds = net(imgs)
        loss  = loss_fn(preds, targets)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        running_loss += loss.item() * imgs.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(loader, net, loss_fn):
    net.eval()
    total_loss = 0.0
    mae_sum    = torch.zeros(NUM_CLASSES, device=device)

    for imgs, targets in loader:
        imgs    = imgs.to(device, non_blocking=True)
        targets = targets.float().to(device, non_blocking=True)

        preds = net(imgs)
        total_loss += loss_fn(preds, targets).item() * imgs.size(0)

        mae_sum += (preds - targets).abs().sum(dim=0)

    avg_loss = total_loss / len(loader.dataset)
    mae      = (mae_sum / len(loader.dataset)).cpu()   # per-class MAE

    return avg_loss, mae


# -----------------  TRAINING DRIVER  -----------------
EPOCHS           = 60
UNFREEZE_EPOCH   = 5
best_val_loss    = float("inf")

for epoch in range(1, EPOCHS + 1):

    # unfreeze backbone once warm-up is done
    if epoch == UNFREEZE_EPOCH + 1:
        for name, p in model.named_parameters():
            if not name.startswith("head."):
                p.requires_grad_(True)
        print("Backbone unfrozen ✅")

    train_loss = train_epoch(train_loader, model, criterion, optimizer, epoch)
    val_loss, val_mae = eval_epoch(test_loader, model, criterion)

    # ---- logging ----
    mae_str = ", ".join([f"{m:.2f}" for m in val_mae])
    print(f"Epoch {epoch:02d} | "
          f"train loss: {train_loss:.4f} | "
          f"val loss: {val_loss:.4f} | "
          f"val MAE/class: [{mae_str}]")

    # save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_choco_count.pt")

print("Training complete.  Best val loss:", best_val_loss)
