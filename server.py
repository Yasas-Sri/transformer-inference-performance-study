import torch


torch.set_num_threads(4)        # intra-op
torch.set_num_interop_threads(1)  


from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel

app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")

model.eval()

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(request: TextRequest):
    inputs = tokenizer(request.text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    return {
        "shape": list(outputs.last_hidden_state.shape)
    }