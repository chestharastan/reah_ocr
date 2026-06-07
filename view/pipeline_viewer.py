#!/usr/bin/env python3
"""
view/pipeline_viewer.py

Walk through every stage of the CNN-BiLSTM-CTC pipeline and show what the
data actually looks like at each step.

Usage
-----
  python view/pipeline_viewer.py [image_path] [--model path.pth] [--num-classes N] [--out out.png]

If no image is given a synthetic text-strip is generated.
If no checkpoint is given random weights are used (shapes are identical; only
the learned pattern inside each feature map will differ).
"""

import argparse, os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image, ImageDraw
import torchvision.transforms as T

# ── path setup ────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "traditional", "src"))
from architectures.cnn_bilstm_ctc import CnnBiLstmCtc


# ═══════════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════════

def make_synthetic_image(width=512, height=64, seed=42):
    """Fake text-line strip (white bg, dark rectangles as character stubs)."""
    img = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(img)
    rng = np.random.default_rng(seed)
    x = 8
    while x < width - 20:
        w = int(rng.uniform(6, 28))
        h = int(rng.uniform(18, 44))
        y = int(rng.uniform(10, height - h - 5))
        fill = int(rng.uniform(0, 50))
        draw.rectangle([x, y, x + w, y + h], fill=fill)
        x += w + int(rng.uniform(2, 10))
    return img


def normalise(arr):
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-8)


def make_channel_grid(fmap, n=8, ncols=4):
    """fmap: (C, H, W) numpy. Returns a tiled canvas of the first n channels."""
    n = min(n, fmap.shape[0])
    nrows = (n + ncols - 1) // ncols
    H, W = fmap.shape[1], fmap.shape[2]
    pad = 2
    canvas = np.zeros((nrows * H + (nrows - 1) * pad,
                        ncols * W + (ncols - 1) * pad))
    for i in range(n):
        r, c = divmod(i, ncols)
        y0, x0 = r * (H + pad), c * (W + pad)
        canvas[y0:y0 + H, x0:x0 + W] = normalise(fmap[i].astype(float))
    return canvas


def fmt_vec(vec, t, n=7):
    nums = "  ".join(f"{v:+.3f}" for v in vec[:n])
    return f"t={t:3d}:  [{nums}  ...]"


# ═══════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", help="Path to input image (optional)")
    ap.add_argument("--model", default=None, help="Path to .pth checkpoint")
    ap.add_argument("--num-classes", type=int, default=105)
    ap.add_argument("--out", default="view/pipeline_output.png")
    args = ap.parse_args()

    nc = args.num_classes

    # ── model ──────────────────────────────────────────────────────────
    model = CnnBiLstmCtc(num_classes=nc)
    if args.model and os.path.exists(args.model):
        ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            state = (ckpt.get("model_state")
                     or ckpt.get("model_state_dict")
                     or ckpt.get("state_dict")
                     or ckpt)
            model.load_state_dict(state)
        print(f"Loaded weights: {args.model}")
    else:
        print("No checkpoint provided — using random weights (shape demo)")
    model.eval()

    # ── image ──────────────────────────────────────────────────────────
    if args.image and os.path.exists(args.image):
        pil = Image.open(args.image).convert("L").resize((512, 64))
        img_label = os.path.basename(args.image)
    else:
        pil = make_synthetic_image()
        img_label = "synthetic text strip"

    tfm = T.Compose([T.Resize((64, 512)), T.ToTensor(), T.Normalize([0.5], [0.5])])
    x = tfm(pil).unsqueeze(0)          # (1, 1, 64, 512)

    # ── forward hooks ──────────────────────────────────────────────────
    # cnn: [Conv2d, BN, ReLU, MaxPool,  Conv2d, BN, ReLU, MaxPool,  Conv2d, BN, ReLU]
    #        0      1   2     3          4      5   6     7          8      9   10
    cap = {}
    hooks = [
        (2,  "conv1_relu"),   # after Conv(1→64) + BN + ReLU
        (3,  "pool1"),        # after MaxPool2d(2,2)
        (6,  "conv2_relu"),   # after Conv(64→128) + BN + ReLU
        (7,  "pool2"),        # after MaxPool2d(2,2)
        (10, "cnn_out"),      # after Conv(128→256) + BN + ReLU  ← CNN done
    ]
    handles = []
    for idx, name in hooks:
        def _hook(m, i, o, _n=name):
            cap[_n] = o.detach()
        handles.append(model.cnn[idx].register_forward_hook(_hook))

    # ── forward pass (step by step to capture seq / lstm) ─────────────
    with torch.no_grad():
        cnn_feat = model.cnn(x)                                    # (1,256,16,128)
        B, C, H, W = cnn_feat.shape
        seq = cnn_feat.permute(0, 3, 1, 2).reshape(B, W, C * H)   # (1,128,4096)
        lstm_out, _ = model.rnn(seq)                               # (1,128,512)
        logits = model.classifier(lstm_out)                        # (1,128,nc)

    for h in handles:
        h.remove()

    # numpy views (squeeze batch dim)
    img_arr  = np.array(pil)
    seq_np   = seq[0].numpy()         # (128, 4096)
    lstm_np  = lstm_out[0].numpy()    # (128, 512)
    logits_np = logits[0].numpy()     # (128, nc)

    # ── colour palette ─────────────────────────────────────────────────
    BG   = "#0d0d0d"
    CELL = "#111827"
    TIT  = "#e2e8f0"
    TXT  = "#86efac"    # green monospace (messy numbers)
    DIM  = "#94a3b8"

    # ── figure / gridspec ──────────────────────────────────────────────
    #   10 rows × 2 cols
    #   CNN rows use the full width (colspan 2);
    #   BiLSTM / logits rows split into [heatmap | text]
    fig = plt.figure(figsize=(18, 40), facecolor=BG)
    gs  = gridspec.GridSpec(
        10, 2,
        figure=fig,
        hspace=0.60, wspace=0.05,
        height_ratios=[1.2,          # 0  input image
                       2.0, 2.0,     # 1–2  conv block 1 (relu + pool)
                       2.0, 2.0,     # 3–4  conv block 2 (relu + pool)
                       2.0,          # 5  conv block 3 (relu, no pool)
                       2.5,          # 6  feature sequence
                       3.0,          # 7  BiLSTM output
                       3.5,          # 8  raw logits
                       2.5],         # 9  CTC decode
    )

    def span_ax(row, title):
        ax = fig.add_subplot(gs[row, 0:2])
        ax.set_facecolor(CELL)
        ax.set_title(title, color=TIT, fontsize=10, fontweight="bold", pad=5)
        for sp in ax.spines.values():
            sp.set_edgecolor("#334155")
        return ax

    def col_ax(row, col, title=None):
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor(CELL)
        if title:
            ax.set_title(title, color=TIT, fontsize=9, fontweight="bold", pad=4)
        for sp in ax.spines.values():
            sp.set_edgecolor("#334155")
        return ax

    def tick_style(ax, fs=7):
        ax.tick_params(colors="#64748b", labelsize=fs)
        ax.xaxis.label.set_color(DIM)
        ax.yaxis.label.set_color(DIM)

    # ── Step 0 — Input Image ───────────────────────────────────────────
    ax0 = span_ax(0, f"Step 0  |  Input Image  ·  {img_label}"
                     "  →  resized to 1 × 64 × 512, normalised to [−1, 1]")
    ax0.imshow(img_arr, cmap="gray", aspect="auto")
    ax0.set_xlabel("pixel column  (0 → 511)", fontsize=8)
    ax0.set_ylabel("row (0→63)", fontsize=8)
    tick_style(ax0)

    # ── Steps 1–5 — CNN blocks (mean map left, 4-channel grid right) ───
    cnn_steps = [
        (1, "conv1_relu", "Step 1  |  Conv2d(1→64, 3×3, pad=1)  +  BatchNorm  +  ReLU",  "64 ch × 64 × 512"),
        (2, "pool1",      "Step 2  |  MaxPool2d(2, 2)  — spatial ÷ 2",                   "64 ch × 32 × 256"),
        (3, "conv2_relu", "Step 3  |  Conv2d(64→128, 3×3, pad=1)  +  BatchNorm  +  ReLU","128 ch × 32 × 256"),
        (4, "pool2",      "Step 4  |  MaxPool2d(2, 2)  — spatial ÷ 2  again",            "128 ch × 16 × 128"),
        (5, "cnn_out",    "Step 5  |  Conv2d(128→256, 3×3, pad=1)  +  BatchNorm  +  ReLU  (no pool)  ← CNN done",
                                                                                           "256 ch × 16 × 128"),
    ]

    for row, key, title, shape_str in cnn_steps:
        fmap = cap[key][0].numpy()      # (C, H, W)
        C, fH, fW = fmap.shape
        mean_map = fmap.mean(axis=0)    # (H, W)  — average activation

        # left col: mean activation map (spatial structure)
        axL = col_ax(row, 0, f"{title}   [{shape_str}]")
        axL.imshow(normalise(mean_map), cmap="viridis", aspect="auto")
        axL.set_xlabel("width  →", fontsize=7)
        axL.set_ylabel("height ↓", fontsize=7)
        tick_style(axL, fs=6)
        axL.text(0.01, 0.03, "mean across all channels",
                 transform=axL.transAxes, fontsize=6, color=DIM,
                 style="italic", va="bottom")

        # right col: 4-channel sample grid
        axR = col_ax(row, 1, f"first 4 individual channels  (of {C})")
        grid = make_channel_grid(fmap, n=4, ncols=4)
        axR.imshow(grid, cmap="viridis", aspect="auto")
        axR.axis("off")

    # ── Step 6 — Feature Sequence ──────────────────────────────────────
    ax6 = span_ax(
        6,
        "Step 6  |  Feature Sequence"
        "  permute(0,3,1,2) → reshape(B, W, C×H)"
        "  →  (1 × 128 time-steps × 4 096 features)"
    )
    # show first 256 feature dims as a heatmap
    im6 = ax6.imshow(seq_np[:, :256].T, cmap="RdBu_r", aspect="auto")
    ax6.set_xlabel("time step  t  (= image column after CNN,  0 → 127)", fontsize=8)
    ax6.set_ylabel("feature dim  (first 256 of 4 096 = 256 ch × 16 rows)", fontsize=8)
    tick_style(ax6)
    fig.colorbar(im6, ax=ax6, fraction=0.012, pad=0.008).ax.tick_params(
        colors="#64748b", labelsize=6)
    ax6.text(0.01, 0.03,
             "each column = one time-step;  its value = all 256 channels stacked vertically at that x-position",
             transform=ax6.transAxes, fontsize=7, color=DIM, style="italic", va="bottom")

    # ── Step 7 — BiLSTM Output ─────────────────────────────────────────
    axL7 = col_ax(7, 0, "Step 7  |  BiLSTM Output  (128 × 512)  — heatmap")
    axR7 = col_ax(7, 1, "sample values  ← messy floats, no direct meaning yet")

    im7 = axL7.imshow(lstm_np.T, cmap="RdBu_r", aspect="auto")
    axL7.set_xlabel("time step", fontsize=8)
    axL7.set_ylabel("hidden dim  (512 = 256 forward + 256 backward)", fontsize=8)
    tick_style(axL7)
    fig.colorbar(im7, ax=axL7, fraction=0.025, pad=0.01).ax.tick_params(
        colors="#64748b", labelsize=6)

    lines7  = [f"  shape:  (128, 512)  =  128 time-steps  ×  512 dims\n",
               f"  hidden = 256-dim forward  +  256-dim backward\n"]
    for t in [0, 1, 2, 30, 63, 64, 100, 126, 127]:
        lines7.append(fmt_vec(lstm_np[t], t, n=6))
    lines7 += ["",
               "  ← each row mixes left-context (→) and",
               "    right-context (←) into one vector",
               "  ← next: Linear(512 → num_classes) maps",
               "    each row to class scores"]
    axR7.text(0.03, 0.97, "\n".join(lines7),
              transform=axR7.transAxes, va="top", fontsize=7.5,
              color=TXT, fontfamily="monospace")
    axR7.axis("off")

    # ── Step 8 — Raw Logits (before CTC) ──────────────────────────────
    axL8 = col_ax(8, 0, f"Step 8  |  Raw Logits before CTC  (128 × {nc})")
    axR8 = col_ax(8, 1, "sample logit rows  ← very messy raw scores")

    im8 = axL8.imshow(logits_np.T, cmap="plasma", aspect="auto")
    axL8.set_xlabel("time step", fontsize=8)
    axL8.set_ylabel(f"class index  (0 = <blank>,  1 … {nc-1} = Khmer chars)", fontsize=8)
    tick_style(axL8)
    fig.colorbar(im8, ax=axL8, fraction=0.025, pad=0.01).ax.tick_params(
        colors="#64748b", labelsize=6)

    argmax_seq = logits[0].argmax(dim=-1).tolist()   # greedy pick per step
    lines8  = [f"  shape:  (128, {nc})\n",
               f"  row = un-normalised score for every class\n"]
    for t in [0, 1, 2, 10, 50, 100, 127]:
        row_v = logits_np[t]
        am    = int(row_v.argmax())
        lines8.append(fmt_vec(row_v, t, n=5) + f"  argmax→ {am}")
    lines8 += ["",
               "  ← no softmax yet — raw linear output",
               f"  ← 0 = <blank>,  1…{nc-1} = characters",
               "  ← CTC loss applies log-softmax internally"]
    axR8.text(0.03, 0.97, "\n".join(lines8),
              transform=axR8.transAxes, va="top", fontsize=7.5,
              color=TXT, fontfamily="monospace")
    axR8.axis("off")

    # ── Step 9 — CTC Greedy Decode ─────────────────────────────────────
    ax9 = span_ax(
        9,
        "Step 9  |  CTC Greedy Decode"
        "  →  argmax per step  →  collapse repeated indices  →  remove blank (0)"
    )

    decoded_ids, prev = [], None
    for idx in argmax_seq:
        if idx != 0 and idx != prev:
            decoded_ids.append(idx)
        prev = idx

    raw_part  = "  ".join(str(i) for i in argmax_seq[:56]) + "  ..."
    coll_part = ("  ".join(str(i) for i in decoded_ids[:56])
                 if decoded_ids else "(all blanks — untrained / no text found)")

    info9 = (
        f"  Raw argmax sequence  (128 steps, first 56 shown):\n"
        f"    {raw_part}\n\n"
        f"  After CTC collapse  →  {len(decoded_ids)} token(s):\n"
        f"    {coll_part}\n\n"
        f"  → map each index to its Khmer character via KhmerVocab → final string"
    )
    ax9.text(0.01, 0.97, info9, transform=ax9.transAxes, va="top",
             fontsize=8.5, color=TXT, fontfamily="monospace")
    ax9.axis("off")

    # ── super-title ────────────────────────────────────────────────────
    plt.suptitle(
        "CNN  ·  BiLSTM  ·  CTC    Pipeline Visualiser    (Khmer OCR)",
        color="white", fontsize=14, fontweight="bold", y=0.997,
    )

    # ── save ───────────────────────────────────────────────────────────
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight", facecolor=BG)
    print(f"Saved  →  {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
