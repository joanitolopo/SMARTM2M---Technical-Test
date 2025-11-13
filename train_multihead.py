# train_multihead.py
"""
Train a multi-head (shared backbone + separate head per component) CNN from scratch using PyTorch.
Dataset format same as before:
  <DATA_DIR>/images/*.png
  <DATA_DIR>/labels/<basename>.json

Each head outputs a single logit (binary open/closed).
Loss = sum of BCEWithLogitsLoss per head (optionally with pos_weight per head).
"""

import os
import random
import json
from glob import glob
from typing import List

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

DATA_DIR = "/content/drive/MyDrive/Non Academic/SMARTM2M - Technical Test/car_dataset_cropped"    # dataset root (images/, labels/)
IMAGE_SIZE = (160, 160)             # input size
BATCH_SIZE = 32
NUM_WORKERS = 4
EPOCHS = 40
LR = 1e-3
WEIGHT_DECAY = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
PATIENCE = 6
CHECKPOINT_PATH = "best_multihead.pth"

LABEL_KEYS = ['front_left', 'front_right', 'rear_left', 'rear_right', 'hood']


POS_WEIGHT = None

# ----------------------------
# Utilities & Seed
# ----------------------------
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ----------------------------
# Dataset
# ----------------------------
class CarComponentsDataset(Dataset):
    def __init__(self, data_dir: str, file_list: List[str], transforms=None):
        self.data_dir = data_dir
        self.images = file_list
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(self.data_dir, "labels", base + ".json")

        img = Image.open(img_path).convert("RGB")
        if self.transforms:
            img = self.transforms(img)

        with open(label_path, 'r') as f:
            label_json = json.load(f)

        # multi-hot vector according to LABEL_KEYS
        label_vec = []
        for k in LABEL_KEYS:
            val = label_json.get(k, "closed")
            label_vec.append(1.0 if str(val).lower().startswith("o") or str(val).lower().startswith("1") else 0.0)

        label = torch.tensor(label_vec, dtype=torch.float32)
        return img, label

# ----------------------------
# Model: Shared backbone + separate heads
# ----------------------------
class MultiHeadCNN(nn.Module):
    def __init__(self, num_heads=len(LABEL_KEYS), backbone_out_features=128, head_hidden=64):
        super().__init__()
        # Shared backbone (same as before)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, backbone_out_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(backbone_out_features),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1))
        )

        # Heads: a small MLP per component producing 1 logit
        self.heads = nn.ModuleDict()
        for i in range(num_heads):
            self.heads[f"head_{i}"] = nn.Sequential(
                nn.Flatten(),
                nn.Linear(backbone_out_features, head_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25),
                nn.Linear(head_hidden, 1)  # single logit
            )

    def forward(self, x):
        feat = self.backbone(x)  # shape (B, C, 1, 1)
        # produce logits per head and stack
        logits_list = []
        for name, head in self.heads.items():
            logits = head(feat)  # shape (B,1)
            logits_list.append(logits.view(-1,1))
        logits = torch.cat(logits_list, dim=1)  # shape (B, num_heads)
        return logits

# ----------------------------
# Metrics helpers
# ----------------------------
def sigmoid_to_binary(preds: torch.Tensor, thresh=0.5):
    return (preds >= thresh).float()

def compute_metrics_logits(logits: torch.Tensor, targets: torch.Tensor, thresh=0.5):
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = sigmoid_to_binary(probs, thresh=thresh)

        preds_np = preds.cpu().numpy()
        targ_np = targets.cpu().numpy()

        eps = 1e-8
        num_labels = preds_np.shape[1]
        precisions = []
        recalls = []
        f1s = []
        supports = []

        for i in range(num_labels):
            p = preds_np[:, i]
            t = targ_np[:, i]
            tp = float(((p == 1) & (t == 1)).sum())
            fp = float(((p == 1) & (t == 0)).sum())
            fn = float(((p == 0) & (t == 1)).sum())
            precision = tp / (tp + fp + eps)
            recall = tp / (tp + fn + eps)
            f1 = 2 * precision * recall / (precision + recall + eps)
            support = int(t.sum())
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            supports.append(support)

        macro_f1 = float(np.mean(f1s))
        return {
            "per_label_precision": precisions,
            "per_label_recall": recalls,
            "per_label_f1": f1s,
            "per_label_support": supports,
            "macro_f1": macro_f1
        }

# ----------------------------
# Training / Validation
# ----------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    n_samples = 0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs)  # shape (B,5)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        bs = imgs.size(0)
        running_loss += loss.item() * bs
        n_samples += bs

    return running_loss / (n_samples + 1e-12)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_samples = 0
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)

            bs = imgs.size(0)
            total_loss += loss.item() * bs
            n_samples += bs

            all_logits.append(logits.cpu())
            all_targets.append(labels.cpu())

    if n_samples == 0:
        return None

    avg_loss = total_loss / (n_samples + 1e-12)
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics_logits(all_logits, all_targets, thresh=0.5)
    return avg_loss, metrics

# ----------------------------
# Data loaders
# ----------------------------
def make_dataloaders(data_dir: str, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, val_frac=0.15, test_frac=0.10):
    imgs = sorted(glob(os.path.join(data_dir, "images", "*.png")))
    if len(imgs) == 0:
        raise RuntimeError(f"No images found in {os.path.join(data_dir, 'images')}")

    random.shuffle(imgs)
    n = len(imgs)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    n_train = n - n_val - n_test

    train_list = imgs[:n_train]
    val_list = imgs[n_train:n_train+n_val]
    test_list = imgs[n_train+n_val:]

    train_tf = T.Compose([
        T.Resize(image_size),
        T.RandomHorizontalFlip(0.5),
        T.ColorJitter(0.12,0.12,0.06),
        T.RandomRotation(8),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    val_tf = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    train_ds = CarComponentsDataset(data_dir, train_list, transforms=train_tf)
    val_ds = CarComponentsDataset(data_dir, val_list, transforms=val_tf)
    test_ds = CarComponentsDataset(data_dir, test_list, transforms=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"Dataset sizes -> train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")
    return train_loader, val_loader, test_loader

# ----------------------------
# Build loss with optional pos_weight per head
# ----------------------------
def make_loss(pos_weight_dict=None, device=DEVICE):
    """
    pos_weight_dict: map label_key -> float (pos weight)
    returns a BCEWithLogitsLoss that works on (B, num_heads) by using pos_weight tensor if provided.
    """
    if pos_weight_dict is None:
        return nn.BCEWithLogitsLoss()
    else:
        pw = []
        for k in LABEL_KEYS:
            pw.append(float(pos_weight_dict.get(k, 1.0)))
        pos_weight_tensor = torch.tensor(pw, dtype=torch.float32, device=device)
        # BCEWithLogitsLoss accepts pos_weight for per-element weighting (applied to positive examples)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

# ----------------------------
# Main training entry
# ----------------------------
def main():
    set_seed(SEED)
    train_loader, val_loader, test_loader = make_dataloaders(DATA_DIR)

    model = MultiHeadCNN(num_heads=len(LABEL_KEYS)).to(DEVICE)
    criterion = make_loss(POS_WEIGHT, device=DEVICE).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float('inf')
    best_macro_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_res = validate(model, val_loader, criterion, DEVICE)
        if val_res is None:
            print("Validation returned None. Stopping.")
            break
        val_loss, val_metrics = val_res

        scheduler.step(val_loss)

        print(f"\nEpoch {epoch}/{EPOCHS}")
        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Val   loss: {val_loss:.4f}")
        print(f"  Val macro F1: {val_metrics['macro_f1']:.4f}")
        for i,k in enumerate(LABEL_KEYS):
            print(f"    {k:12s} -> P:{val_metrics['per_label_precision'][i]:.3f} R:{val_metrics['per_label_recall'][i]:.3f} F1:{val_metrics['per_label_f1'][i]:.3f} sup:{val_metrics['per_label_support'][i]}")

        improved = False
        if val_metrics['macro_f1'] > best_macro_f1 or val_loss < best_val_loss:
            improved = True
            best_macro_f1 = max(best_macro_f1, val_metrics['macro_f1'])
            best_val_loss = min(best_val_loss, val_loss)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics
            }, CHECKPOINT_PATH)
            print(f"  ✅ Saved checkpoint to {CHECKPOINT_PATH}")

        if improved:
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  No improvement. patience {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    # Final test evaluation
    print("\nLoading best checkpoint and evaluating on test set...")
    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        test_res = validate(model, test_loader, criterion, DEVICE)
        if test_res is not None:
            test_loss, test_metrics = test_res
            print(f"Test loss: {test_loss:.4f}")
            print(f"Test macro F1: {test_metrics['macro_f1']:.4f}")
            for i,k in enumerate(LABEL_KEYS):
                print(f"  {k:12s} -> P:{test_metrics['per_label_precision'][i]:.3f} R:{test_metrics['per_label_recall'][i]:.3f} F1:{test_metrics['per_label_f1'][i]:.3f} sup:{test_metrics['per_label_support'][i]}")
        else:
            print("Test set empty or evaluation error.")
    else:
        print("No checkpoint found to evaluate.")

    print("\nTraining complete.")

if __name__ == "__main__":
    main()