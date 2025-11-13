# SMARTM2M — Technical Test  

**Realtime Car Component State Detection (Open/Close)**  

---

## 🚗 Project Overview

The goal is to build a fully working end-to-end system that can:

- Detect **five car components** (Open / Closed):
  - Front Left Door  
  - Front Right Door  
  - Rear Left Door  
  - Rear Right Door  
  - Hood
- Perform detection **in realtime from the browser**, using live images captured directly from the user's current web session.
- Train a model **from scratch (no transfer learning)**.
- Produce a **single-page UI widget** showing the AI predicted status.

This repository contains:
- Dataset collection automation  
- PyTorch model + training  
- Inference FastAPI WebSocket server  
- Realtime browser widget (`realtime_infer_widget.js`)  
- Evaluation + threshold tuning utilities  

You can reproduce the entire pipeline end-to-end.

---

## 🧰 Repository Structure

```
├── car_dataset_cropped/          # Captured dataset (images + labels)
│   ├── images/
│   └── labels/
├── best_multihead.pth            # Best model checkpoint
├── full_data_collector.py        # Automated dataset collection
├── train_multihead.py            # PyTorch model training
├── predict.py                    # Model loading + inference helper
├── server.py                     # FastAPI WebSocket inference server
├── client_send.py                # Simple Python WebSocket test client
├── realtime_infer_widget.js      # Browser realtime inference UI widget
├── export_model.py               # Export to ONNX + TorchScript
├── requirements.txt              # Python dependencies
└── README.md
```

---

## 🔧 Installation

### 1. Clone repository

```bash
git clone https://github.com/joanitolopo/SMARTM2M---Technical-Test.git
cd SMARTM2M---Technical-Test
```

### 2. Create virtual environment & install dependencies

```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

---

## 📸 Dataset Collection

The dataset is captured automatically from the 3D car webpage.

**Run:**

```bash
python full_data_collector.py
```

The script will:
- Load the car webpage
- Toggle all 32 combinations of open/closed component states
- Capture multiple angles
- Save images and JSON labels

**Output folder:**

```bash
car_dataset_cropped/images/
car_dataset_cropped/labels/
```

---

## 🧠 Model Training

Train the multi-head CNN:

```bash
python train_multihead.py
```

- Saves best checkpoint → `best_multihead.pth`
- Uses 5 independent binary heads
- Supports pos‐weight & full augmentation
- Automatically logs accuracy/loss per head

---

## 🚀 Inference Server (WebSocket)

Start realtime inference backend:

```bash
python server.py
```

**Runs on:**

```bash
ws://localhost:8000/ws/infer
```

**Client should send:**

```json
{ "image": "data:image/png;base64,..." }
```

**Server returns:**

```json
{
  "result": {
    "front_left":  { "prob": 0.92, "pred": 1 },
    "front_right": { "prob": 0.03, "pred": 0 },
    "rear_left":   { "prob": 0.87, "pred": 1 },
    "rear_right":  { "prob": 0.75, "pred": 1 },
    "hood":        { "prob": 0.04, "pred": 0 }
  }
}
```

**Test using:**

```bash
python client_send.py path/to/image.png
```

---

## 🖥 Realtime Web Integration

This is one of the most important parts of the test. The model must read the real canvas from the webpage the user is currently using.

This repository includes a full browser widget:

```
realtime_infer_widget.js
```

### How to run it:

#### Step 1 — Open the car simulation webpage

Example:

```
https://euphonious-concha-ab5c5d.netlify.app/
```

#### Step 2 — Open Developer Tools

Press:

```
F12  → Console tab
```

#### Step 3 — Copy–paste the entire script

Open this file:

```
realtime_infer_widget.js
```

Copy its entire content → paste into the browser console → press Enter.

#### Step 4 — Click "Connect" on the widget

A floating UI will appear:

```
+-----------------------------------------+
| AI realtime detected state              |
| WS: connected                           |
| [Connect] [Disconnect] [Capture]        |
|-----------------------------------------|
| Front Left   →  OPEN (0.98)             |
| Front Right  →  CLOSED (0.03)           |
| Rear Left    →  OPEN (0.91)             |
| Rear Right   →  OPEN (0.76)             |
| Hood         →  CLOSED (0.05)           |
+-----------------------------------------+
```

#### Step 5 — Interact with car buttons

The widget will automatically:
- Detect button click
- Wait for animation
- Capture canvas only
- Send image to WebSocket inference server
- Update the predicted state

### Notes

If your backend runs on another machine / IP:

```js
window.CUSTOM_WS_HOST = "http://YOUR_IP:8000";
```

---

## 📦 Export Model (Optional)

```bash
python export_model.py
```

Produces:
- `model.onnx`
- `model_ts.pt` (TorchScript)

---

## 📊 Results & Threshold Tuning

To compute probabilities & optimize thresholds:

```bash
python compute_logits_and_labels.py
python tune_thresholds.py
```

This helps improve precision/recall for each head independently.

---

## 🛣 Roadmap (Optional Improvements)

- Add temporal smoothing for realtime stability
- Collect dataset under more lighting conditions
- Build a polished full UI instead of injected console widget
- Deploy backend with HTTPS and WSS support

---

## 📝 License

This project is provided for the SMARTM2M technical test.  
Code is free for demonstration and evaluation purposes.