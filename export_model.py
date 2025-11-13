# export_models.py
import torch
from predict import load_model, LABEL_KEYS
import onnx

MODEL_PATH = "best_multihead.pth"
ONNX_PATH = "model.onnx"
TS_PATH = "model_ts.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def export_onnx():
    model = load_model(MODEL_PATH, device=DEVICE)
    model.eval()
    dummy = torch.randn(1,3,160,160, device=DEVICE)
    torch.onnx.export(model, dummy, ONNX_PATH, opset_version=12,
                      input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input":{0:"batch"}, "logits":{0:"batch"}})
    print("ONNX saved to", ONNX_PATH)

def export_torchscript():
    model = load_model(MODEL_PATH, device=DEVICE)
    model.eval()
    example = torch.randn(1,3,160,160, device=DEVICE)
    traced = torch.jit.trace(model, example)
    traced.save(TS_PATH)
    print("TorchScript saved to", TS_PATH)

if __name__ == "__main__":
    export_onnx()
    export_torchscript()
