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






| Backend        | Workers            | Requests/sec (RPS) | Avg Latency | p95 Latency |
|---------------|-------------------|--------------------|------------|------------|
| PyTorch       | 4 (thread tuned)  | 88                 | 456 ms     | 1.39 s     |
| ONNX (Python) | 1                 | 121                | 307 ms     | 613 ms     |
| ONNX (Python) | 4                 | 116                | 326 ms     | 676 ms     |