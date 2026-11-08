## Task 3 results: context length

### Standard evaluation (as logged in metrics.json: non-overlapping windows)

Seed noise floor (median std over configs): 0.0036 bpb → a gain counts as real if > 2×noise = 0.0072

| model | seq_len | val bpb (mean ± std) | gain vs previous seq_len | real gain? | seeds | train tokens |
|---|---:|---:|---:|---|---:|---:|
| Transformer, modern (4×256, 3.2M params) | 64 | **2.1562** ± 0.0006 | – | – | 2 | 16.4M |
| Transformer, modern (4×256, 3.2M params) | 128 | **2.1135** ± 0.0090 | +0.0427 | yes | 2 | 16.4M |
| Transformer, modern (4×256, 3.2M params) | 256 | **2.1158** ± 0.0010 | -0.0023 | **no** | 2 | 16.4M |
| Transformer, modern (4×256, 3.2M params) | 512 | **2.1030** ± 0.0007 | +0.0129 | yes | 2 | 16.4M |
| LSTM 2×256 (1.1M params) | 64 | **2.1806** ± 0.0051 | – | – | 2 | 16.4M |
| LSTM 2×256 (1.1M params) | 128 | **2.1488** ± 0.0028 | +0.0318 | yes | 2 | 16.4M |
| LSTM 2×256 (1.1M params) | 256 | **2.1372** ± 0.0045 | +0.0116 | yes | 2 | 16.4M |
| LSTM 2×256 (1.1M params) | 512 | **2.1294** ± 0.0058 | +0.0078 | yes | 2 | 16.4M |

### Sliding-window evaluation (stride = seq_len/8: every scored character has ≥ 7/8 of the window as context)

Seed noise floor (median std over configs): 0.0027 bpb → a gain counts as real if > 2×noise = 0.0055

| model | seq_len | val bpb (mean ± std) | gain vs previous seq_len | real gain? | seeds | train tokens |
|---|---:|---:|---:|---|---:|---:|
| Transformer, modern (4×256, 3.2M params) | 64 | **2.0917** ± 0.0004 | – | – | 2 | 16.4M |
| Transformer, modern (4×256, 3.2M params) | 128 | **2.0781** ± 0.0095 | +0.0136 | yes | 2 | 16.4M |
| Transformer, modern (4×256, 3.2M params) | 256 | **2.0970** ± 0.0011 | -0.0189 | **no** | 2 | 16.4M |
| Transformer, modern (4×256, 3.2M params) | 512 | **2.0916** ± 0.0018 | +0.0054 | **no** | 2 | 16.4M |
| LSTM 2×256 (1.1M params) | 64 | **2.1141** ± 0.0037 | – | – | 2 | 16.4M |
| LSTM 2×256 (1.1M params) | 128 | **2.1110** ± 0.0016 | +0.0031 | **no** | 2 | 16.4M |
| LSTM 2×256 (1.1M params) | 256 | **2.1155** ± 0.0050 | -0.0045 | **no** | 2 | 16.4M |
| LSTM 2×256 (1.1M params) | 512 | **2.1166** ± 0.0060 | -0.0010 | **no** | 2 | 16.4M |

Mean bits/char by position bucket (seq_len=512 models, 2 seeds):

| model | pos 1–4 | pos 5–16 | pos 17–64 | pos 65–128 | pos 129–256 | pos 257–512 |
|---|---:|---:|---:|---:|---:|---:|
| Transformer, modern (4×256, 3.2M params) | 3.075 | 2.239 | 2.118 | 2.073 | 2.103 | 2.086 |
| LSTM 2×256 (1.1M params) | 3.350 | 2.278 | 2.153 | 2.102 | 2.117 | 2.112 |
