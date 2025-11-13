# server.py
import base64
import io
import json
import asyncio
from typing import Dict
from PIL import Image
import numpy as np

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn

from predict import load_model, predict_image, LABEL_KEYS  # reuse

MODEL_PATH = "best_multihead.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = 0.5
EMA_ALPHA = 0.4 

app = FastAPI()
model = load_model(MODEL_PATH, DEVICE)

# Keep EMA state per websocket connection
class ConnectionManager:
    def __init__(self):
        self.active = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active[websocket] = {"ema": None}

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active:
            self.active.pop(websocket, None)

manager = ConnectionManager()

def decode_base64_image(data_b64: str):
    header, b64 = (data_b64.split(",",1) + [None])[:2] if "," in data_b64 else (None,data_b64)
    img_bytes = base64.b64decode(b64)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")

@app.websocket("/ws/infer")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    print("Client connected")
    try:
        while True:
            msg = await websocket.receive_text()
            # expecting JSON: {"image": "data:image/png;base64,...."}
            j = json.loads(msg)
            img_b64 = j.get("image")
            if not img_b64:
                await websocket.send_text(json.dumps({"error":"no image"}))
                continue
            pil = decode_base64_image(img_b64)
            preds = predict_image(pil, model, device=DEVICE, threshold=THRESHOLD)  # dict
            # build probs vector in order
            probs = np.array([preds[k]['prob'] for k in LABEL_KEYS], dtype=float)

            # EMA smoothing
            state = manager.active[websocket]
            if state["ema"] is None:
                state["ema"] = probs
            else:
                state["ema"] = EMA_ALPHA * probs + (1-EMA_ALPHA) * state["ema"]
            smooth_probs = state["ema"]
            smooth_preds = (smooth_probs >= THRESHOLD).astype(int)

            out = {k: {"prob": float(smooth_probs[i]), "pred": int(smooth_preds[i])} for i,k in enumerate(LABEL_KEYS)}
            await websocket.send_text(json.dumps({"result": out}))
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("Client disconnected")
    except Exception as e:
        manager.disconnect(websocket)
        print("Error in websocket loop:", e)

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
