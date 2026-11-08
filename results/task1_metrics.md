## Task 1a: metrics recomputed by hand (trained char LSTM)

Run: `outputs/nanolm/2026-11-03_15-00-34_hw1_t1_trained_char_lstm` (lstm, 1,069,824 params, 16,384,000 training tokens)

Inputs from metrics.json: val_loss = 1.493252 nats/token, val_tokens = 111,488, val_bytes = 111,488

| quantity | formula | by hand | logged | abs. diff |
|---|---|---:|---:|---:|
| loss (nats/token) | mean −ln p | 1.493252 | 1.493252 | 0 |
| perplexity | e^loss | 4.451548 | 4.451548 | 0.00e+00 |
| bits/token | loss / ln 2 | 2.154307 | – | – |
| bits-per-byte | loss · tokens / (bytes · ln 2) | 2.154307 | 2.154307 | 0.00e+00 |

Total information in the validation text: 240,179 bits for 111,488 bytes (a plain 8-bit file would need 891,904 bits → compression ratio 3.71×).

## Task 1b: untrained models vs ln V

`init` = evaluated at random initialization (train.lr=0, so no weight changes); `1 step` = the assignment's `train.max_steps=1` (a single AdamW step at the first, tiny warm-up learning rate).

| tokenizer | V | ln V | state | val loss | loss − ln V | ppl | bytes/token | bpb |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| char | 65 | 4.1744 | init | 4.1970 | +0.0226 | 66.5 | 1.00 | 6.0550 |
| char | 65 | 4.1744 | 1 step | 4.1787 | +0.0043 | 65.3 | 1.00 | 6.0286 |
| gpt2 | 50,257 | 10.8249 | init | 10.8232 | -0.0017 | 50,172.6 | 3.09 | 5.0489 |
| gpt2 | 50,257 | 10.8249 | 1 step | 10.8035 | -0.0214 | 49,192.3 | 3.09 | 5.0397 |

A model that predicts every token with probability 1/V has loss exactly −ln(1/V) = ln V and perplexity exactly V; random initialization is close to that uniform guess.
