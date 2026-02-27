import time
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

model_path = "distilbert.onnx"


so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession(model_path, so)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

text = "Let the dragon rides again on the winds of time"
inputs = tokenizer(text, return_tensors="np")


for _ in range(10):
    session.run(None, {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"]
    })


runs = 100
start = time.time()

for _ in range(runs):
    session.run(None, {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"]
    })

end = time.time()

avg_latency = (end - start) / runs * 1000
print(f"Average ONNX latency: {avg_latency:.2f} ms")