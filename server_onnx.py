import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer

app = FastAPI()

# Configure ONNX session
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession("distilbert.onnx", so)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(request: TextRequest):
    inputs = tokenizer(request.text, return_tensors="np")

    outputs = session.run(
        None,
        {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        },
    )

    return {"shape": list(outputs[0].shape)}