"""HW1 Task 3 -- context length: bpb vs train.seq_len for the LSTM and the modern Transformer,
plus a per-position loss analysis that shows how much past context each model actually uses.

Reads the sweep produced by  python -m nanolm.sweep --sweep nanolm/configs/sweeps/hw1_context.yaml
Run from the nanolm `llm/` directory:
    python $HW/scripts/task3_context.py --sweep outputs/nanolm/sweeps/hw1_context --out $HW
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")  # run from llm/ so `nanolm` is importable
from nanolm.data import read_shard  # noqa: E402
from nanolm.generate import load_run  # noqa: E402

LN2 = math.log(2)
COLOR = {"gpt": "#2a78d6", "lstm": "#eb6834"}
LABEL = {"gpt": "Transformer, modern (4×256, 3.2M params)", "lstm": "LSTM 2×256 (1.1M params)"}


def load_metrics(sweep: Path) -> pd.DataFrame:
    rows = []
    for d in sorted(sweep.iterdir()):
        f = d / "metrics.json"
        if not f.exists():
            continue
        m = json.loads(f.read_text())
        arch, ctx = re.match(r"(gpt|lstm)_ctx(\d+)", m["name"]).groups()
        # sliding-window bpb from `python -m nanolm.compress <run> --bits --strides 0 <seq_len/8>`
        slide = float("nan")
        if (d / "compress.json").exists():
            st = [s for s in json.loads((d / "compress.json").read_text())["stride"] if s["stride"] > 0]
            slide = st[0]["bpb"] if st else slide
        rows.append({"arch": arch, "ctx": int(ctx), "seed": m["seed"], "bpb": m["val_bpb"], "bpb_slide": slide,
                     "tokens": m["tokens_seen"], "params": m["params"], "time_s": m["train_time_sec"],
                     "run_dir": str(d)})
    return pd.DataFrame(rows)


def summary_table(df: pd.DataFrame, col: str = "bpb", title: str = "") -> tuple[str, pd.DataFrame]:
    g = df.groupby(["arch", "ctx"])[col].agg(["mean", "std", "count"]).reset_index()
    noise = df.groupby(["arch", "ctx"])[col].std().median()  # seed noise floor
    out = [f"### {title}\n",
           f"Seed noise floor (median std over configs): {noise:.4f} bpb → a gain counts as real if > 2×noise = "
           f"{2 * noise:.4f}\n",
           "| model | seq_len | val bpb (mean ± std) | gain vs previous seq_len | real gain? | seeds | train tokens |",
           "|---|---:|---:|---:|---|---:|---:|"]
    for arch in ["gpt", "lstm"]:
        sub = g[g.arch == arch].sort_values("ctx")
        prev = None
        for r in sub.itertuples():
            tok = int(df[(df.arch == arch) & (df.ctx == r.ctx)].tokens.iloc[0])
            if prev is None:
                gain, real = "–", "–"
            else:
                d = prev - r.mean
                gain, real = f"{d:+.4f}", ("yes" if d > 2 * noise else "**no**")
            out.append(f"| {LABEL[arch]} | {r.ctx} | **{r.mean:.4f}** ± {r.std:.4f} | {gain} | {real} | "
                       f"{r.count} | {tok / 1e6:.1f}M |")
            prev = r.mean
    return "\n".join(out), g


def fig_bpb_vs_ctx(g: pd.DataFrame, gs: pd.DataFrame, df: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for arch in ["gpt", "lstm"]:
        for grp, col, ls, mk, tag in [(g, "bpb", "-", "o", "standard eval (non-overlapping windows)"),
                                      (gs, "bpb_slide", "--", "s", "sliding eval (≥ 7/8 of window as context)")]:
            s = grp[grp.arch == arch].sort_values("ctx")
            ax.errorbar(s.ctx, s["mean"], yerr=s["std"], color=COLOR[arch], marker=mk, ls=ls, lw=2, capsize=4,
                        label=f"{LABEL[arch].split(' (')[0]} — {tag}", alpha=1 if ls == "-" else 0.8)
            pts = df[df.arch == arch]
            ax.scatter(pts.ctx, pts[col], color=COLOR[arch], s=12, alpha=0.4, zorder=3)
            for r in s.itertuples():
                ax.annotate(f"{r.mean:.3f}", (r.ctx, r.mean), textcoords="offset points", xytext=(6, 5),
                            fontsize=7.5, color=COLOR[arch])
    ax.set_xscale("log", base=2)
    ax.set_xticks([64, 128, 256, 512])
    ax.set_xticklabels(["64", "128", "256", "512"])
    ax.set_xlabel("context length train.seq_len (characters, log scale)")
    ax.set_ylabel("validation bits-per-byte (lower is better)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5)
    ax.set_title("Task 3: bpb vs context length (same 16.4M-token budget per run;\n"
                 "line = mean of 2 seeds, error bar = std, dots = individual seeds)", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def per_position_loss(run_dir: str, T: int, device: str) -> np.ndarray:
    """Mean cross-entropy (nats) at each position 0..T-1 inside non-overlapping validation windows.
    Position t has seen t+1 characters of context, so this curve shows how loss falls with available context."""
    cfg, model, tok, dinfo, meta = load_run(run_dir, device)
    dataset = json.loads((Path(run_dir) / "metrics.json").read_text())["dataset"]
    val = np.asarray(read_shard(Path(cfg.data.cache_dir) / dataset / "val.bin"), dtype=np.int64)
    n = (len(val) - 1) // T
    x = torch.from_numpy(val[: n * T].reshape(n, T))
    y = torch.from_numpy(val[1: n * T + 1].reshape(n, T))
    total = torch.zeros(T, dtype=torch.float64)
    for i in range(0, n, 16):
        xb, yb = x[i:i + 16].to(dinfo.device), y[i:i + 16].to(dinfo.device)
        logits = model(xb).float()
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1), reduction="none").view(xb.shape)
        total += ce.sum(0).cpu().double()  # MPS has no float64
    return (total / n).numpy()


def fig_per_position(df: pd.DataFrame, path: Path, device: str, ctx: int = 512) -> str:
    edges = [0, 4, 16, 64, 128, 256, 512]
    curves, rows = {}, []
    for arch in ["gpt", "lstm"]:
        runs = df[(df.arch == arch) & (df.ctx == ctx)].run_dir.tolist()
        c = np.mean([per_position_loss(r, ctx, device) for r in runs], axis=0) / LN2
        curves[arch] = c
        rows.append((arch, [c[a:b].mean() for a, b in zip(edges[:-1], edges[1:])]))
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    pos = np.arange(1, ctx + 1)
    for arch, c in curves.items():
        sm = pd.Series(c).rolling(9, center=True, min_periods=1).mean()
        ax.plot(pos, c, color=COLOR[arch], alpha=0.25, lw=0.8)
        ax.plot(pos, sm, color=COLOR[arch], lw=2, label=LABEL[arch])
    ax.set_xscale("log", base=2)
    ax.set_xlabel("characters of context available (position in the window + 1, log scale)")
    ax.set_ylabel("validation bits per character at that position")
    ax.set_ylim(1.8, 4.6)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5)
    ax.set_title(f"Task 3: how loss falls with available context (seq_len={ctx} models, mean of 2 seeds)",
                 loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    head = "| model | " + " | ".join(f"pos {a + 1}–{b}" for a, b in zip(edges[:-1], edges[1:])) + " |"
    lines = [f"Mean bits/char by position bucket (seq_len={ctx} models, 2 seeds):\n", head,
             "|---|" + "---:|" * (len(edges) - 1)]
    for arch, vals in rows:
        lines.append(f"| {LABEL[arch]} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="outputs/nanolm/sweeps/hw1_context")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "results").mkdir(parents=True, exist_ok=True)
    df = load_metrics(Path(a.sweep))
    table, g = summary_table(df, "bpb", "Standard evaluation (as logged in metrics.json: non-overlapping windows)")
    table_s, gs = summary_table(df, "bpb_slide", "Sliding-window evaluation (stride = seq_len/8: every scored "
                                                 "character has ≥ 7/8 of the window as context)")
    fig_bpb_vs_ctx(g, gs, df, out / "figures" / "task3_bpb_vs_context.png")
    pos_table = fig_per_position(df, out / "figures" / "task3_loss_vs_position.png", a.device)
    text = ("## Task 3 results: context length\n\n" + table + "\n\n" + table_s + "\n\n" + pos_table + "\n")
    print(text)
    (out / "results" / "task3_results.md").write_text(text)
    print("[saved] results/task3_results.md, figures/task3_bpb_vs_context.png, figures/task3_loss_vs_position.png")


if __name__ == "__main__":
    main()
