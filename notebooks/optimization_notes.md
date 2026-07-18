# Inference Optimization Results

## Executive summary

- Baseline exact-match accuracy: **79.50%**.
- Dynamic INT8 exact-match accuracy: **79.50%**, so it caused no measured accuracy loss.
- **FP32 on MPS produced the best forward result:** 6.746 ms median latency, a **1.65x speedup** over the FP32 CPU baseline.
- MPS provided only a small generation improvement: 18.065 ms versus 18.726 ms, a **1.04x speedup**.
- `torch.compile` did not provide a meaningful improvement on CPU or MPS.
- BF16 was slower than FP32 on both CPU and MPS.
- Dynamic INT8 was approximately **2.5x slower** than FP32 CPU.

## Current recommendation

Use **eager FP32 on MPS** for batch-64 forward inference. Keep BF16, dynamic INT8, and `torch.compile` disabled for this workload. CPU FP32 and MPS FP32 are nearly tied for generation, so test both at the actual deployment batch size.

The tests used batch size 64. Forward used a 14-token sequence. Generation used an 8-token prompt and generated 7 tokens. Speedups below use **median latency**:

## Speedup comparison

| Configuration | Forward median | Forward speedup | Generate median | Generate speedup | Exact-match accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP32 CPU baseline | 11.137 ms | 1.00x | 18.726 ms | 1.00x | 79.50% |
| FP32 CPU + `inference_mode` | 11.516 ms | 0.97x | 20.713 ms | 0.90x | Not recorded |
| FP32 CPU + `torch.compile` | 11.473 ms | 0.97x | 20.015 ms | 0.94x | Not recorded |
| BF16 CPU | 38.248 ms | 0.29x | 49.617 ms | 0.38x | Not recorded |
| **FP32 MPS** | **6.746 ms** | **1.65x** | 18.065 ms | 1.04x | Not recorded |
| FP32 MPS + `torch.compile` | 6.745 ms | 1.65x | **17.953 ms** | **1.04x** | Not recorded |
| BF16 MPS | 7.401 ms | 1.50x | 19.542 ms | 0.96x | Not recorded |
| Dynamic INT8 CPU | 28.536 ms | 0.39x | 46.728 ms | 0.40x | 79.50% |
| Dynamic INT8 CPU + `torch.compile` | 25.759 ms | 0.43x | 46.727 ms | 0.40x | Not recorded |

The 0.001 ms forward difference between eager and compiled MPS, and the 0.112 ms generation difference, are too small to treat as meaningful given the observed variability. Eager FP32 MPS is therefore preferred.

## Complete forward results

All forward tests used batch size 64 and sequence length 14. Token rate is input tokens per second.

| Configuration | Mean ms | Std ms | Median ms | P95 ms | Calls/s | Examples/s | Input tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FP32 CPU baseline | 11.741 | 1.556 | 11.137 | 15.485 | 85.17 | 5,450.88 | 76,312.31 |
| FP32 CPU + `inference_mode` | 11.738 | 0.799 | 11.516 | 13.063 | 85.20 | 5,452.53 | 76,335.35 |
| FP32 CPU + `torch.compile` | 13.936 | 4.774 | 11.473 | 20.605 | 71.76 | 4,592.43 | 64,293.97 |
| BF16 CPU | 40.752 | 4.581 | 38.248 | 49.219 | 24.54 | 1,570.49 | 21,986.83 |
| **FP32 MPS** | **6.664** | **0.231** | **6.746** | **6.897** | **150.06** | **9,604.04** | **134,456.50** |
| FP32 MPS + `torch.compile` | 6.685 | 0.255 | 6.745 | 6.932 | 149.59 | 9,573.91 | 134,034.67 |
| BF16 MPS | 7.373 | 0.371 | 7.401 | 7.930 | 135.64 | 8,680.68 | 121,529.58 |
| Dynamic INT8 CPU | 30.172 | 4.791 | 28.536 | 39.476 | 33.14 | 2,121.20 | 29,696.77 |
| Dynamic INT8 CPU + `torch.compile` | 27.409 | 5.364 | 25.759 | 35.098 | 36.48 | 2,334.98 | 32,689.68 |

## Complete generation results

All generation tests used batch size 64, an 8-token prompt, and 7 generated tokens. Token rate is generated tokens per second.

| Configuration | Mean ms | Std ms | Median ms | P95 ms | Calls/s | Examples/s | Generated tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FP32 CPU baseline | 18.770 | 0.634 | 18.726 | 19.856 | 53.28 | 3,409.78 | 23,868.49 |
| FP32 CPU + `inference_mode` | 23.056 | 13.745 | 20.713 | 25.545 | 43.37 | 2,775.84 | 19,430.85 |
| FP32 CPU + `torch.compile` | 21.123 | 3.367 | 20.015 | 29.511 | 47.34 | 3,029.89 | 21,209.22 |
| BF16 CPU | 49.652 | 0.401 | 49.617 | 50.316 | 20.14 | 1,288.97 | 9,022.79 |
| FP32 MPS | 18.656 | 2.300 | 18.065 | 22.764 | 53.60 | 3,430.48 | 24,013.33 |
| FP32 MPS + `torch.compile` | 20.250 | 4.363 | 17.953 | 28.514 | 49.38 | 3,160.55 | 22,123.85 |
| BF16 MPS | 19.609 | 0.484 | 19.542 | 20.521 | 51.00 | 3,263.80 | 22,846.57 |
| Dynamic INT8 CPU | 48.201 | 4.194 | 46.728 | 59.959 | 20.75 | 1,327.78 | 9,294.43 |
| Dynamic INT8 CPU + `torch.compile` | 46.937 | 1.720 | 46.727 | 48.092 | 21.31 | 1,363.54 | 9,544.79 |
