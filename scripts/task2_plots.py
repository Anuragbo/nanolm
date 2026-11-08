"""HW1 Task 2 -- RNN vs GRU vs LSTM vs Transformer: learning curves, overfitting figure, results table.

Reads the sweep produced by  python -m nanolm.sweep --sweep nanolm/configs/sweeps/01_char_models.yaml
Run from the nanolm `llm/` directory:
    python $HW/scripts/task2_plots.py --sweep outputs/nanolm/sweeps/char_models --out $HW
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LN2 = math.log(2)
ORDER = ["gpt_modern", "lstm", "gru", "lstm_512", "gpt_textbook", "rnn"]
LABEL = {
    "gpt_modern": "Transformer, modern (RMSNorm+RoPE+SwiGLU)",
    "gpt_textbook": "Transformer, GPT-2 style",
    "lstm": "LSTM 2×256",
    "lstm_512": "LSTM 2×512",
    "gru": "GRU 2×256",
    "rnn": "vanilla RNN 2×256",
}
COLOR = {"gpt_modern": "#2a78d6", "lstm": "#eb6834", "gru": "#1baf7a", "lstm_512": "#eda100",
         "gpt_textbook": "#6250d6", "rnn": "#e34948"}


def read_jsonl(p: Path) -> pd.DataFrame:
    return pd.DataFrame([json.loads(l) for l in p.read_text().splitlines() if l.strip()])


def load(sweep: Path):
    metrics, evals, trains = [], [], []
    for d in sorted(sweep.iterdir()):
        if not (d / "metrics.json").exists():
            continue
        m = json.loads((d / "metrics.json").read_text())
        name = m["name"]
        metrics.append(m)
        e = read_jsonl(d / "eval_log.jsonl").assign(name=name, seed=m["seed"])
        t = read_jsonl(d / "train_log.jsonl").assign(name=name, seed=m["seed"])
        tok_per_step = m["tokens_seen"] / m["steps"]
        t["train_tokens"] = t["step"] * tok_per_step
        evals.append(e)
        trains.append(t)
    return pd.DataFrame(metrics), pd.concat(evals), pd.concat(trains)


def results_table(M: pd.DataFrame, E: pd.DataFrame) -> str:
    rows = []
    for n in [x for x in ORDER if x in set(M.name)]:
        m = M[M.name == n]
        e = E[E.name == n].groupby("step")["bpb"].mean()
        rows.append({
            "model": LABEL[n], "params": int(m.params.iloc[0]),
            "bpb_mean": m.val_bpb.mean(), "bpb_std": m.val_bpb.std(ddof=1) if len(m) > 1 else float("nan"),
            "best_bpb": e.min(), "best_step": int(e.idxmin()), "final_loss": m.val_loss.mean(),
            "seeds": len(m), "tokens": int(m.tokens_seen.iloc[0]), "time_s": m.train_time_sec.mean(),
        })
    df = pd.DataFrame(rows).sort_values("bpb_mean")
    out = ["| model | params | val bpb (mean ± std) | val loss (nats) | best val bpb (step) | seeds | train tokens | train time |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in df.itertuples():
        out.append(f"| {r.model} | {r.params / 1e6:.2f}M | **{r.bpb_mean:.3f}** ± {r.bpb_std:.3f} | {r.final_loss:.3f} | "
                   f"{r.best_bpb:.3f} ({r.best_step}) | {r.seeds} | {r.tokens / 1e6:.1f}M | {r.time_s:.0f} s |")
    return "\n".join(out)


def fig_curves(E: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for n in [x for x in ORDER if x in set(E.name)]:
        g = E[E.name == n].groupby("train_tokens")["bpb"]
        mean, lo, hi = g.mean(), g.min(), g.max()
        x = mean.index / 1e6
        ax.plot(x, mean.values, color=COLOR[n], lw=2, label=LABEL[n])
        ax.fill_between(x, lo.values, hi.values, color=COLOR[n], alpha=0.2, lw=0)
    ax.set_xlabel("training tokens seen (millions of characters)")
    ax.set_ylabel("validation bits-per-byte (lower is better)")
    ax.set_ylim(2.0, 3.0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5)
    ax.set_title("Task 2: RNN vs GRU vs LSTM vs Transformer on Tiny Shakespeare\n"
                 "(same 16.4M-token budget; line = mean of 2 seeds, band = seed range)", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_overfit(E: pd.DataFrame, T: pd.DataFrame, path: Path, names=("lstm", "lstm_512", "rnn")):
    names = [n for n in names if n in set(E.name)]
    fig, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, n in zip(axes, names):
        t = T[T.name == n].groupby("train_tokens")["loss"].mean()
        t = t.rolling(5, min_periods=1).mean()  # smooth the noisy per-batch loss
        e = E[E.name == n].groupby("train_tokens")["loss"].mean()
        ax.plot(t.index / 1e6, t.values / LN2, color="#898781", lw=1.5, label="train (per-batch, smoothed)")
        ax.plot(e.index / 1e6, e.values / LN2, color=COLOR[n], lw=2.2, marker="o", ms=3, label="validation")
        best = e.idxmin()
        ax.axvline(best / 1e6, color=COLOR[n], ls=":", lw=1)
        ax.annotate(f"best val {e.min() / LN2:.3f}\n@ {best / 1e6:.1f}M tok", (best / 1e6, e.min() / LN2),
                    textcoords="offset points", xytext=(5, 18), fontsize=8)
        gap = (e.iloc[-1] - t.iloc[-1]) / LN2
        ax.set_title(f"{LABEL[n]}\nfinal val − train gap = {gap:+.3f} bits/char", loc="left", fontsize=9.5)
        ax.set_xlabel("training tokens seen (millions)")
        ax.grid(alpha=0.3)
        ax.set_ylim(1.2, 3.2)
    axes[0].set_ylabel("bits per character (loss / ln 2)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Task 2: train vs validation — does the bigger LSTM overfit? (mean of 2 seeds; "
                 "train loss includes dropout)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="outputs/nanolm/sweeps/char_models")
    ap.add_argument("--out", required=True, help="hw1 folder (writes figures/ and results/)")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "results").mkdir(parents=True, exist_ok=True)
    M, E, T = load(Path(a.sweep))
    table = results_table(M, E)
    print(table)
    (out / "results" / "task2_results.md").write_text("## Task 2 results\n\n" + table + "\n")
    fig_curves(E, out / "figures" / "task2_learning_curves.png")
    fig_overfit(E, T, out / "figures" / "task2_train_vs_val.png")
    print("[saved] results/task2_results.md, figures/task2_learning_curves.png, figures/task2_train_vs_val.png")


if __name__ == "__main__":
    main()
