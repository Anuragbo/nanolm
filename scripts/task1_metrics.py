"""HW1 Task 1 -- recompute loss / perplexity / bits-per-token / bits-per-byte by hand from metrics.json,
and compare the loss of untrained models with ln V.

Run from the nanolm `llm/` directory:
    python $HW/scripts/task1_metrics.py --out $HW/results/task1_metrics.md
"""
import argparse
import json
import math
from pathlib import Path

LN2 = math.log(2)


def latest(pattern: str) -> Path:
    runs = sorted(Path("outputs/nanolm").glob(f"*_{pattern}"))
    if not runs:
        raise SystemExit(f"no run matching *_{pattern} under outputs/nanolm")
    return runs[-1]


def load(run: Path) -> dict:
    m = json.loads((run / "metrics.json").read_text())
    meta = json.loads((Path("data/nanolm") / m["dataset"] / "meta.json").read_text())
    m["vocab_size"] = meta["vocab_size"]
    return m


def by_hand(m: dict) -> dict:
    loss = m["val_loss"]  # mean nats per token
    return {
        "ppl": math.exp(loss),
        "bits_per_token": loss / LN2,
        "bpb": loss * m["val_tokens"] / (m["val_bytes"] * LN2),
        "total_bits": loss * m["val_tokens"] / LN2,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    lines = []
    p = lines.append

    run = latest("hw1_t1_trained_char_lstm")
    m = load(run)
    h = by_hand(m)
    p("## Task 1a: metrics recomputed by hand (trained char LSTM)\n")
    p(f"Run: `{run}` ({m['arch']}, {m['params']:,} params, {m['tokens_seen']:,} training tokens)\n")
    p(f"Inputs from metrics.json: val_loss = {m['val_loss']:.6f} nats/token, "
      f"val_tokens = {m['val_tokens']:,}, val_bytes = {m['val_bytes']:,}\n")
    p("| quantity | formula | by hand | logged | abs. diff |")
    p("|---|---|---:|---:|---:|")
    p(f"| loss (nats/token) | mean −ln p | {m['val_loss']:.6f} | {m['val_loss']:.6f} | 0 |")
    p(f"| perplexity | e^loss | {h['ppl']:.6f} | {m['val_ppl']:.6f} | {abs(h['ppl'] - m['val_ppl']):.2e} |")
    p(f"| bits/token | loss / ln 2 | {h['bits_per_token']:.6f} | – | – |")
    p(f"| bits-per-byte | loss · tokens / (bytes · ln 2) | {h['bpb']:.6f} | {m['val_bpb']:.6f} | "
      f"{abs(h['bpb'] - m['val_bpb']):.2e} |")
    p(f"\nTotal information in the validation text: {h['total_bits']:,.0f} bits for {m['val_bytes']:,} bytes "
      f"(a plain 8-bit file would need {8 * m['val_bytes']:,} bits → compression ratio {8 / h['bpb']:.2f}×).\n")

    p("## Task 1b: untrained models vs ln V\n")
    p("`init` = evaluated at random initialization (train.lr=0, so no weight changes); "
      "`1 step` = the assignment's `train.max_steps=1` (a single AdamW step at the first, tiny warm-up learning rate).\n")
    p("| tokenizer | V | ln V | state | val loss | loss − ln V | ppl | bytes/token | bpb |")
    p("|---|---:|---:|---|---:|---:|---:|---:|---:|")
    for tok in ["char", "gpt2"]:
        for state, tag in [("init", f"hw1_t1_init_{tok}"), ("1 step", f"hw1_t1_step1_{tok}")]:
            m = load(latest(tag))
            lnv = math.log(m["vocab_size"])
            p(f"| {tok} | {m['vocab_size']:,} | {lnv:.4f} | {state} | {m['val_loss']:.4f} | "
              f"{m['val_loss'] - lnv:+.4f} | {m['val_ppl']:,.1f} | {m['val_bytes_per_token']:.2f} | {m['val_bpb']:.4f} |")
    p("\nA model that predicts every token with probability 1/V has loss exactly −ln(1/V) = ln V "
      "and perplexity exactly V; random initialization is close to that uniform guess.")

    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
        print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
