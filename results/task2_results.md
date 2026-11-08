## Task 2 results

| model | params | val bpb (mean ± std) | val loss (nats) | best val bpb (step) | seeds | train tokens | train time |
|---|---:|---:|---:|---:|---:|---:|---:|
| Transformer, modern (RMSNorm+RoPE+SwiGLU) | 3.16M | **2.116** ± 0.001 | 1.467 | 2.116 (2000) | 2 | 16.4M | 190 s |
| LSTM 2×256 | 1.07M | **2.137** ± 0.005 | 1.481 | 2.137 (2000) | 2 | 16.4M | 68 s |
| GRU 2×256 | 0.81M | **2.160** ± 0.002 | 1.497 | 2.160 (2000) | 2 | 16.4M | 276 s |
| LSTM 2×512 | 4.24M | **2.170** ± 0.008 | 1.504 | 2.128 (1000) | 2 | 16.4M | 154 s |
| Transformer, GPT-2 style | 3.44M | **2.185** ± 0.002 | 1.515 | 2.185 (2000) | 2 | 16.4M | 140 s |
| vanilla RNN 2×256 | 0.28M | **2.437** ± 0.006 | 1.689 | 2.437 (2000) | 2 | 16.4M | 165 s |
