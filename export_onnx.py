import torch
from transformers import AutoTokenizer, AutoModel


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")
model.eval()


text = "Let the dragon rides again on the winds of time"
inputs = tokenizer(text, return_tensors="pt")


torch.onnx.export(
    model,
    (inputs["input_ids"], inputs["attention_mask"]),
    "distilbert.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["last_hidden_state"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
    },
    opset_version=14
)

