This project investigates CPU-based Transformer inference performance 
using DistilBERT. It benchmarks and compares:

- PyTorch eager inference
- ONNX Runtime optimized inference
- Multi-worker process scaling
- Thread-level tuning
- Load testing under concurrent traffic (k6)



Environment:
- 12th Gen Intel i7-12700H
- Arch Linux
- CPU-only inference
