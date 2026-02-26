import time
import torch
from transformers import AutoTokenizer, AutoModel


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")

model.eval()

text = "Let the dragon rides again on the winds of time"

inputs = tokenizer(text,return_tensors="pt")

for _ in range(5):
    with torch.no_grad():
        _ = model(**inputs)

start = time.time()

with torch.no_grad():
     outptuts = model(**inputs)

end  =time.time()

print(f"Single inference time: {(end - start) * 1000:.2f} ms")

        
