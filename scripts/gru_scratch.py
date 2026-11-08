"""HW1 extra credit -- a GRU written gate by gate (in the style of nanolm's LSTMScratch),
verified numerically against PyTorch's nn.GRU.

PyTorch's nn.GRU computes, for each layer and time step (gate order in the weights: r, z, n):
    r  = sigma(W_ir x + b_ir + W_hr h + b_hr)          reset gate   -- how much of the past to use
    z  = sigma(W_iz x + b_iz + W_hz h + b_hz)          update gate  -- how much of the old state to keep
    n  = tanh (W_in x + b_in + r * (W_hn h + b_hn))    candidate state (note: r multiplies W_hn h + b_hn)
    h' = (1 - z) * n + z * h                           <- interpolation: when z ~ 1 the state (and its
                                                          gradient) passes through unchanged, like the LSTM cell

Checks (all against nn.GRU with the SAME weights, 2 layers, batch_first):
  1. forward: outputs and final hidden states, float64 and float32 (CPU, and MPS if available)
  2. backward: gradients w.r.t. the input and every weight/bias
  3. end to end: train two identical char-level LMs on Tiny Shakespeare (one with nn.GRU, one with
     GRUScratch) for 30 AdamW steps from the same init; their loss curves must coincide

Run from the nanolm `llm/` directory:
    python $HW/scripts/gru_scratch.py --out $HW/results/gru_check.txt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUCellScratch(nn.Module):
    """One GRU layer, one time step at a time (for reading, not for speed)."""

    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.hidden = hidden
        # Kept separate: the reset gate multiplies only the hidden part of the candidate (W_hn h + b_hn).
        self.W_x = nn.Linear(in_dim, 3 * hidden)  # [W_ir; W_iz; W_in]
        self.W_h = nn.Linear(hidden, 3 * hidden)  # [W_hr; W_hz; W_hn]

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None):
        B, T, _ = x.shape
        h = x.new_zeros(B, self.hidden) if h0 is None else h0
        outs = []
        for t in range(T):
            xr, xz, xn = self.W_x(x[:, t]).chunk(3, dim=-1)
            hr, hz, hn = self.W_h(h).chunk(3, dim=-1)
            r = torch.sigmoid(xr + hr)
            z = torch.sigmoid(xz + hz)
            n = torch.tanh(xn + r * hn)
            h = (1 - z) * n + z * h
            outs.append(h)
        return torch.stack(outs, dim=1), h


class GRUScratch(nn.Module):
    """Multi-layer GRU with the same call signature as nn.GRU(batch_first=True) (no dropout)."""

    def __init__(self, in_dim: int, hidden: int, num_layers: int = 1):
        super().__init__()
        self.layers = nn.ModuleList([GRUCellScratch(in_dim if i == 0 else hidden, hidden)
                                     for i in range(num_layers)])

    def forward(self, x):
        finals = []
        for layer in self.layers:
            x, h = layer(x)
            finals.append(h)
        return x, torch.stack(finals, dim=0)

    @torch.no_grad()
    def load_from_nn_gru(self, ref: nn.GRU):
        for i, layer in enumerate(self.layers):
            layer.W_x.weight.copy_(getattr(ref, f"weight_ih_l{i}"))
            layer.W_x.bias.copy_(getattr(ref, f"bias_ih_l{i}"))
            layer.W_h.weight.copy_(getattr(ref, f"weight_hh_l{i}"))
            layer.W_h.bias.copy_(getattr(ref, f"bias_hh_l{i}"))
        return self

    def grads_like_nn_gru(self):
        """Gradients renamed to nn.GRU's parameter names, for a one-to-one comparison."""
        g = {}
        for i, layer in enumerate(self.layers):
            g[f"weight_ih_l{i}"] = layer.W_x.weight.grad
            g[f"bias_ih_l{i}"] = layer.W_x.bias.grad
            g[f"weight_hh_l{i}"] = layer.W_h.weight.grad
            g[f"bias_hh_l{i}"] = layer.W_h.bias.grad
        return g


def compare(dtype, device, D=64, H=128, L=2, B=8, T=50, seed=0):
    torch.manual_seed(seed)
    ref = nn.GRU(D, H, num_layers=L, batch_first=True).to(device, dtype)
    mine = GRUScratch(D, H, L).to(device, dtype).load_from_nn_gru(ref)
    x = torch.randn(B, T, D, device=device, dtype=dtype)
    x1, x2 = x.clone().requires_grad_(), x.clone().requires_grad_()
    y_ref, h_ref = ref(x1)
    y_me, h_me = mine(x2)
    w = torch.randn_like(y_ref)
    ((y_ref * w).sum() + h_ref.pow(2).sum()).backward()
    ((y_me * w).sum() + h_me.pow(2).sum()).backward()
    def d(a, b):
        ab = (a - b).abs().max().item()
        return ab, ab / max(a.abs().max().item(), 1e-30)
    diffs = {"output": d(y_ref, y_me), "final hidden": d(h_ref, h_me), "grad input": d(x1.grad, x2.grad)}
    mg = mine.grads_like_nn_gru()
    per_param = [d(p.grad, mg[n]) for n, p in ref.named_parameters()]
    diffs["grad weights (worst param)"] = max(per_param, key=lambda t: t[1])
    return diffs


class TinyLM(nn.Module):
    def __init__(self, V, D, rnn):
        super().__init__()
        self.emb, self.rnn, self.head = nn.Embedding(V, D), rnn, nn.Linear(D, V)

    def forward(self, idx):
        return self.head(self.rnn(self.emb(idx))[0])


def train_compare(steps=30, D=64, L=2, B=16, T=64):
    """Same init, same batches, same optimizer: nn.GRU vs GRUScratch inside a char LM (float64, CPU)."""
    sys.path.insert(0, ".")
    from nanolm.data import read_shard
    data = np.asarray(read_shard("data/nanolm/tinyshakespeare_char_tail100/train.bin"), dtype=np.int64)
    V = 65
    torch.manual_seed(0)
    ref = TinyLM(V, D, nn.GRU(D, D, num_layers=L, batch_first=True)).double()
    mine = TinyLM(V, D, GRUScratch(D, D, L)).double()
    mine.emb.load_state_dict(ref.emb.state_dict())
    mine.head.load_state_dict(ref.head.state_dict())
    mine.rnn.load_from_nn_gru(ref.rnn)
    opts = [torch.optim.AdamW(m.parameters(), lr=3e-3) for m in (ref, mine)]
    rng = np.random.default_rng(0)
    rows = []
    for step in range(1, steps + 1):
        ix = rng.integers(0, len(data) - T - 1, B)
        x = torch.from_numpy(np.stack([data[i:i + T] for i in ix]))
        y = torch.from_numpy(np.stack([data[i + 1:i + T + 1] for i in ix]))
        losses = []
        for m, opt in zip((ref, mine), opts):
            loss = F.cross_entropy(m(x).reshape(-1, V), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        rows.append((step, *losses))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    lines = ["HW1 extra credit: GRU written gate by gate (GRUScratch) vs torch.nn.GRU",
             f"torch {torch.__version__}; 2 layers, input 64, hidden 128, batch 8, 50 time steps, identical weights", ""]
    tol = {torch.float64: 1e-10, torch.float32: 1e-5}
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    all_ok = True
    for device in devices:
        for dtype in ([torch.float64, torch.float32] if device == "cpu" else [torch.float32]):
            diffs = compare(dtype, device)
            ok = all(rel < tol[dtype] for _, rel in diffs.values())
            all_ok &= ok
            lines.append(f"[{device} {str(dtype).split('.')[-1]}]  relative tolerance {tol[dtype]:.0e}  -> "
                         f"{'PASS' if ok else 'FAIL'}")
            for k, (ab, rel) in diffs.items():
                lines.append(f"    {k:<27s} max |diff| {ab:.3e}   relative {rel:.3e}")
            lines.append("")
    rows = train_compare()
    max_gap = max(abs(r[1] - r[2]) for r in rows)
    ok = max_gap < 1e-8
    all_ok &= ok
    lines.append(f"[end-to-end training, float64 CPU] char LM on Tiny Shakespeare, 30 AdamW steps from the same init "
                 f"-> {'PASS' if ok else 'FAIL'} (max loss gap {max_gap:.2e})")
    lines.append("    step   loss nn.GRU        loss GRUScratch")
    for s, l1, l2 in rows:
        if s in (1, 2, 5, 10, 20, 30):
            lines.append(f"    {s:4d}   {l1:.12f}   {l2:.12f}")
    lines.append("")
    lines.append(f"OVERALL: {'PASS' if all_ok else 'FAIL'}")
    text = "\n".join(lines)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n")
        print(f"\n[saved] {a.out}")


if __name__ == "__main__":
    main()
