from fastapi import FastAPI

app = FastAPI(title="RF-WorldPose Edge/Cloud Inference", version="0.1.0")

@app.get("/health")
def health():
    return {"status": "ok", "runtime": "onnxruntime-stub"}

@app.post("/predict")
def predict(payload: dict):
    # TODO: online preprocess -> ONNX Runtime/Triton client -> confidence gate.
    return {"prediction": "unknown", "confidence": 0.0, "model_version": "stub"}
