"""HW1 Task 4 -- attention analysis: quantify what each head of the trained char Transformer looks at.

For every layer/head and every text, it measures the average attention weight (over query positions) on:
  prev       the previous character (t-1)                                     -> previous-token heads
  self       the character itself (t)
  word_start the first character of the current word (only queries where that is >= 2 steps back,
             so it never overlaps with `prev`)                                -> word-start heads
  line_start the most recent newline / ':' / '.' (sentence or speaker-line marker; queries where it is >= 2 back)
  induction  (repeated text only) the character AFTER the earlier copy of the current character,
             i.e. key t-P+1 for a query t in the second copy (period P)        -> induction / copy heads
  first      position 0 (an "attention sink")
and the mean look-back distance sum_s A[t,s]*(t-s).
Each score is compared with a uniform-attention baseline (mean 1/(t+1) over the same queries).

Run from the nanolm `llm/` directory:
    python $HW/scripts/task4_heads.py --run outputs/nanolm/sweeps/char_models/gpt_modern_s1337 --out $HW
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, ".")
from nanolm.viz import attention_maps, load_run  # noqa: E402

TEXTS = {
    "shakespeare": "ROMEO:\nBut soft! what light through yonder window breaks?\nIt is the east, and Juliet is the sun.",
    "modern_prose": "The researchers trained a small model on a laptop, and it learned to spell words before grammar.",
    "repeated": "the quick brown fox jumps over the lazy dog. the quick brown fox jumps over the lazy dog.",
}
MARKERS = set("\n:.")


def word_start(text: str, t: int) -> int:
    s = t
    while s > 0 and text[s - 1].isalpha():
        s -= 1
    return s


def last_marker(text: str, t: int) -> int | None:
    for s in range(t, -1, -1):
        if text[s] in MARKERS:
            return s
    return None


def score_heads(A: np.ndarray, text: str, period: int | None) -> dict:
    """A: [H, T, T] attention of one layer. Returns {metric: (per-head scores [H], uniform baseline, n_queries)}."""
    H, T, _ = A.shape
    out = {}

    def avg(pairs):  # pairs = [(query t, key s)]
        if not pairs:
            return None
        sc = np.array([[A[h, t, s] for t, s in pairs] for h in range(H)]).mean(1)
        base = float(np.mean([1.0 / (t + 1) for t, _ in pairs]))
        return sc, base, len(pairs)

    out["prev"] = avg([(t, t - 1) for t in range(1, T)])
    out["self"] = avg([(t, t) for t in range(T)])
    out["word_start"] = avg([(t, word_start(text, t)) for t in range(T)
                             if text[t].isalpha() and word_start(text, t) <= t - 2])
    out["line_start"] = avg([(t, m) for t in range(T) if (m := last_marker(text, t)) is not None and m <= t - 2])
    out["first"] = avg([(t, 0) for t in range(1, T)])
    if period:
        out["induction"] = avg([(t, t - period + 1) for t in range(period, T) if t - period + 1 < t - 1])
    dist = np.array([[(A[h, t, : t + 1] * (t - np.arange(t + 1))).sum() for t in range(T)] for h in range(H)]).mean(1)
    out["lookback"] = (dist, float(np.mean([t / 2 for t in range(T)])), T)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="outputs/nanolm/sweeps/char_models/gpt_modern_s1337")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    cfg, model, tok, _, _ = load_run(a.run)
    metrics = ["prev", "self", "word_start", "line_start", "induction", "first", "lookback"]
    all_scores = {}  # (text, metric) -> [L, H] array
    lines = ["## Task 4: attention head analysis\n",
             f"Model: `{a.run}` ({cfg.model.n_layers} layers × {cfg.model.n_heads} heads, "
             f"{cfg.model.norm}/{cfg.model.pos}/{cfg.model.mlp}). Score = average attention weight on the named "
             f"position over all applicable query positions; `uniform` = the same average for evenly spread "
             f"attention (1/(t+1)). `lookback` = mean distance (characters) of the attended keys.\n"]
    for name, text in TEXTS.items():
        assert all(c in tok.stoi for c in text), f"{name}: character outside the 65-char vocabulary"
        idx = torch.tensor([tok.encode(text)])
        maps = attention_maps(model, idx)  # list of [H, T, T]
        assert all(np.allclose(m.sum(-1), 1, atol=1e-4) for m in maps)
        period = text.index(". ") + 2 if name == "repeated" else None  # start of the second copy
        per_layer = [score_heads(m, text, period) for m in maps]
        L, H = len(maps), maps[0].shape[0]
        lines.append(f"\n### Text: {name} ({len(text)} chars)\n\n```\n{text}\n```\n")
        hdr = [m for m in metrics if per_layer[0].get(m) is not None]
        lines.append("| head | " + " | ".join(hdr) + " |")
        lines.append("|---|" + "---:|" * len(hdr))
        for l in range(L):
            for h in range(H):
                cells = []
                for m in hdr:
                    sc, base, n = per_layer[l][m]
                    v = sc[h]
                    fmt = f"{v:.1f}" if m == "lookback" else f"{v:.2f}"
                    strong = (m != "lookback" and v >= 0.3 and v >= 3 * base)
                    cells.append(f"**{fmt}**" if strong else fmt)
                lines.append(f"| L{l}·H{h} | " + " | ".join(cells) + " |")
        base_cells = []
        for m in hdr:
            _, base, n = per_layer[0][m]
            base_cells.append(f"{base:.1f}" if m == "lookback" else f"{base:.2f} (n={n})")
        lines.append("| *uniform* | " + " | ".join(base_cells) + " |")
        for m in hdr:
            all_scores[(name, m)] = np.array([per_layer[l][m][0] for l in range(L)])

    lines.append("\n### Head types (best head per behaviour; bold in the tables = score ≥ 0.30 and ≥ 3× uniform)\n")
    lines.append("| behaviour | head | " + " | ".join(TEXTS) + " |")
    lines.append("|---|---|" + "---:|" * len(TEXTS))
    for m in ["prev", "word_start", "line_start", "induction", "self", "first"]:
        texts = [t for t in TEXTS if (t, m) in all_scores]
        mean = np.mean([all_scores[(t, m)] for t in texts], axis=0)
        l, h = np.unravel_index(mean.argmax(), mean.shape)
        vals = [f"{all_scores[(t, m)][l, h]:.2f}" if (t, m) in all_scores else "–" for t in TEXTS]
        lines.append(f"| {m} | L{l}·H{h} | " + " | ".join(vals) + " |")
    text = "\n".join(lines)
    (out / "results" / "task4_heads.md").write_text(text + "\n")
    print(text)

    show = ["prev", "word_start", "line_start", "induction", "self", "first"]
    fig, axes = plt.subplots(1, len(show), figsize=(2.9 * len(show), 3.2))
    for ax, m in zip(axes, show):
        texts = [t for t in TEXTS if (t, m) in all_scores]
        M = np.mean([all_scores[(t, m)] for t in texts], axis=0)
        ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
        for l in range(M.shape[0]):
            for h in range(M.shape[1]):
                ax.text(h, l, f"{M[l, h]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if M[l, h] > 0.55 else "black")
        ax.set_xticks(range(M.shape[1]))
        ax.set_yticks(range(M.shape[0]))
        ax.set_xlabel("head")
        ax.set_ylabel("layer")
        ax.set_title(m + ("" if len(texts) == len(TEXTS) else f"\n({', '.join(texts)})"), fontsize=9)
    fig.suptitle("Task 4: average attention weight on the relevant position, per layer × head "
                 "(mean over texts)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "figures" / "task4_head_scores.png", dpi=150, bbox_inches="tight")
    print("[saved] results/task4_heads.md, figures/task4_head_scores.png")


if __name__ == "__main__":
    main()
