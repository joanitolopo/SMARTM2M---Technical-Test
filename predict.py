# predict.py
import os
import json
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np

# config 
MODEL_PATH = "best_multihead.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_KEYS = ['front_left','front_right','rear_left','rear_right','hood']
IMAGE_SIZE = (160,160)

import torch.nn as nn
class MultiHeadCNN(nn.Module):
    def __init__(self, num_heads=5, backbone_out_features=128, head_hidden=64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, backbone_out_features, 3, 1, 1),
            nn.BatchNorm2d(backbone_out_features), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.heads = nn.ModuleList()
        for _ in range(num_heads):
            self.heads.append(nn.Sequential(
                nn.Flatten(),
                nn.Linear(backbone_out_features, head_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25),
                nn.Linear(head_hidden, 1)
            ))
    def forward(self,x):
        feat = self.backbone(x)
        outs = [h(feat).view(-1,1) for h in self.heads]
        return torch.cat(outs, dim=1)

import re
import torch

def load_model(path=MODEL_PATH, device=DEVICE):
    """
    Load MultiHeadCNN model from checkpoint `path` (robust to naming differences).
    - Supports checkpoints saved as {'model_state_dict': state_dict, ...} or raw state_dict.
    - Auto-remaps head key names 'heads.head_0.*' <-> 'heads.0.*' depending on which present.
    """
    model = MultiHeadCNN(num_heads=len(LABEL_KEYS))
    # load checkpoint
    ckpt = torch.load(path, map_location=device)

    # extract state_dict
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        sd = ckpt['model_state_dict']
    elif isinstance(ckpt, dict) and all(k.startswith('module.') for k in ckpt.keys()):
        # sometimes whole module saved; assume it's state_dict-like
        sd = ckpt
    elif isinstance(ckpt, dict) and any(key.startswith('heads.head_') for key in ckpt.keys()):
        sd = ckpt
    else:
        # ckpt might be a state_dict already
        sd = ckpt

    model_sd_keys = set(model.state_dict().keys())
    ckpt_keys = set(sd.keys())

    # quick checks for patterns
    has_heads_dotnum = any(re.match(r'^heads\.\d+\.', k) for k in ckpt_keys)
    has_heads_headnum = any(re.match(r'^heads\.head_\d+\.', k) for k in ckpt_keys)
    model_has_dotnum = any(re.match(r'^heads\.\d+\.', k) for k in model_sd_keys)
    model_has_headnum = any(re.match(r'^heads\.head_\d+\.', k) for k in model_sd_keys)

    remapped_sd = sd  # default

    # Case A: ckpt uses heads.head_N but model expects heads.N -> remap 'heads.head_0.' -> 'heads.0.'
    if has_heads_headnum and model_has_dotnum:
        remapped = {}
        for k,v in sd.items():
            new_k = re.sub(r'^heads\.head_(\d+)\.', r'heads.\1.', k)
            remapped[new_k] = v
        remapped_sd = remapped
        print("[load_model] remapped checkpoint keys: heads.head_N -> heads.N")

    # Case B: ckpt uses heads.N but model expects heads.head_N -> remap 'heads.0.' -> 'heads.head_0.'
    elif has_heads_dotnum and model_has_headnum:
        remapped = {}
        for k,v in sd.items():
            new_k = re.sub(r'^heads\.(\d+)\.', r'heads.head_\1.', k)
            remapped[new_k] = v
        remapped_sd = remapped
        print("[load_model] remapped checkpoint keys: heads.N -> heads.head_N")

    try:
        model.load_state_dict(remapped_sd)
        print(f"[load_model] Loaded state_dict into model (strict=True).")
    except RuntimeError as e:
        print("[load_model] strict load failed:", e)
        try:
            model.load_state_dict(remapped_sd, strict=False)
            print("[load_model] Loaded state_dict into model with strict=False (some keys ignored).")
        except Exception as e2:
            print("[load_model] Failed to load state_dict even with strict=False:", e2)
            raise e2

    model.to(device).eval()
    return model


# preprocessing
preprocess = T.Compose([
    T.Resize(IMAGE_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def predict_image(img: Image.Image, model, device=DEVICE, threshold=0.5):
    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()[0]  # shape (5,)
    preds = (probs >= threshold).astype(int)
    result = {k: {'prob': float(p),'pred': int(pred)} for k,pred,k,p in zip(preds, probs, LABEL_KEYS, probs)}
    res = {}
    for i,k in enumerate(LABEL_KEYS):
        res[k] = {'prob': float(probs[i]), 'pred': int(preds[i])}
    return res

# quick CLI test
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict.py path/to/image.png")
        sys.exit(1)
    path = sys.argv[1]
    img = Image.open(path).convert("RGB")
    model = load_model()
    out = predict_image(img, model)
    print(json.dumps(out, indent=2))
