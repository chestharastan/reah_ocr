"""
Unified inference for every trained Khmer-OCR architecture.

Each trained run lives at:
    <outputs_root>/<run_name>/checkpoints/best_model.pth
    <outputs_root>/<run_name>/checkpoints/experiment.json   <- holds the full config

This script:
  1. Discovers every trained run under --outputs-root (defaults to ../outputs_crnn).
  2. Rebuilds each model from its own saved config via the architecture REGISTRY,
     so it works for ALL architectures (CTC and attention) automatically.
  3. Runs prediction on one or more images / a folder of images.
  4. Prints a table and optionally writes a CSV.

CLI examples
------------
    # Predict one image with every trained model
    python final/predict.py path/to/word.png

    # Predict every image in a folder, save results
    python final/predict.py path/to/images_dir --save final/predictions.csv

    # Only run specific runs
    python final/predict.py img.png --models vgg_bilstm_ctc_10k vgg_bilstm_ctc_50k
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

# --------------------------------------------------------------------------- #
# Make the project's `src/` importable (vocab, architectures registry, ...).
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import architectures  # noqa: E402  (builds REGISTRY + build_model / build_transform)
from vocab import KhmerVocab, KhmerVocabAttention  # noqa: E402

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_OUTPUTS_ROOT = REPO_ROOT / "outputs_crnn"
DEFAULT_CHARSET = REPO_ROOT / "charset.json"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def is_attention(config: dict) -> bool:
    """Attention models decode autoregressively with <sos>/<eos>."""
    arch = config["model"]["architecture"]
    return config["model"].get("decoder") == "attention" or "attention" in arch


def extract_state_dict(checkpoint):
    """Pull the raw state_dict out of the various checkpoint wrapper formats."""
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unexpected checkpoint type: {type(checkpoint)}")
    for key in ("model_state", "model_state_dict", "state_dict", "model"):
        if key in checkpoint:
            return checkpoint[key]
    return checkpoint  # assume the dict itself is the state dict


def infer_num_classes(state_dict):
    """Read num_classes from the classifier weight (works for CTC + attention)."""
    for key, tensor in state_dict.items():
        if "classifier" in key and key.endswith("weight"):
            return tensor.shape[0]
    return None


def decode_attention(ids, vocab: KhmerVocabAttention) -> str:
    """Greedy decode for attention models: stop at <eos>, drop <sos>/<blank>."""
    chars = []
    for idx in ids:
        if idx == vocab.eos_id:
            break
        if idx in (0, vocab.sos_id):
            continue
        chars.append(vocab.idx_to_char[idx])
    return "".join(chars)


# --------------------------------------------------------------------------- #
# Predictor: one trained model, rebuilt from its own config.
# --------------------------------------------------------------------------- #
class Predictor:
    def __init__(self, run_name, config, checkpoint_path, charset_path, device):
        self.run_name = run_name
        self.arch = config["model"]["architecture"]
        self.attention = is_attention(config)
        self.device = device

        # Vocab (attention needs the extended <sos>/<eos> variant)
        VocabCls = KhmerVocabAttention if self.attention else KhmerVocab
        self.vocab = VocabCls(str(charset_path))

        # Load weights and rebuild the model from the registry.
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Some checkpoints store the entire model object.
        if hasattr(checkpoint, "eval"):
            self.model = checkpoint.to(device).eval()
        else:
            state_dict = extract_state_dict(checkpoint)
            num_classes = infer_num_classes(state_dict) or len(self.vocab)
            self.model = architectures.build_model(config, num_classes)
            self.model.load_state_dict(state_dict)
            self.model.to(device).eval()

        self.transform = architectures.build_transform(config)

    @torch.no_grad()
    def predict(self, image_path) -> str:
        # Match training preprocessing: load as single-channel grayscale.
        img = Image.open(image_path).convert("L")
        x = self.transform(img).unsqueeze(0).to(self.device)
        logits = self.model(x)              # (1, T, num_classes) for both families
        ids = logits.squeeze(0).argmax(dim=-1).tolist()
        if self.attention:
            return decode_attention(ids, self.vocab)
        return self.vocab.decode(ids)


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def discover_runs(outputs_root):
    """Find every trained run: (run_name, config, checkpoint_path)."""
    runs = []
    for exp_json in sorted(Path(outputs_root).rglob("experiment.json")):
        ckpt_dir = exp_json.parent
        checkpoint = ckpt_dir / "best_model.pth"
        if not checkpoint.exists():
            checkpoint = ckpt_dir / "last_model.pth"
        if not checkpoint.exists():
            continue
        with open(exp_json) as f:
            config = json.load(f)["config"]
        run_name = ckpt_dir.parent.name  # <run_name>/checkpoints/experiment.json
        runs.append((run_name, config, checkpoint))
    return runs


def gather_images(paths):
    """Expand files / directories into a flat sorted list of image paths."""
    images = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            images += sorted(q for q in p.rglob("*") if q.suffix.lower() in IMAGE_EXTS)
        elif p.suffix.lower() in IMAGE_EXTS:
            images.append(p)
        else:
            print(f"  (skipping non-image: {p})")
    return images


def resolve_charset(config, default_charset):
    """Prefer the charset the run was trained with; fall back to the default."""
    run_charset = config.get("dataset", {}).get("charset")
    if run_charset and Path(run_charset).exists():
        return run_charset
    return default_charset


def load_predictors(outputs_root, charset_path, device, only=None):
    """Build a Predictor for every discovered run (optionally filtered by name)."""
    predictors = []
    for run_name, config, checkpoint in discover_runs(outputs_root):
        if only and run_name not in only:
            continue
        try:
            charset = resolve_charset(config, charset_path)
            predictors.append(Predictor(run_name, config, checkpoint, charset, device))
            print(f"  loaded  {run_name:<32} ({config['model']['architecture']})")
        except Exception as e:  # one bad checkpoint shouldn't kill the rest
            print(f"  FAILED  {run_name:<32} {type(e).__name__}: {e}")
    return predictors


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Khmer OCR inference across all architectures")
    parser.add_argument("images", nargs="+", help="Image file(s) and/or folder(s)")
    parser.add_argument("--outputs-root", default=str(DEFAULT_OUTPUTS_ROOT),
                        help="Root dir holding trained runs (default: ../outputs_crnn)")
    parser.add_argument("--charset", default=str(DEFAULT_CHARSET),
                        help="Path to charset.json")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Only run these run names (default: all discovered)")
    parser.add_argument("--save", default=None, help="Optional CSV output path")
    parser.add_argument("--device", default=None, help="cuda / cpu (auto by default)")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    print(f"\nDiscovering models under {args.outputs_root} ...")
    predictors = load_predictors(args.outputs_root, args.charset, device, only=args.models)
    if not predictors:
        print("No models found. Check --outputs-root / --models.")
        sys.exit(1)

    images = gather_images(args.images)
    if not images:
        print("No images found.")
        sys.exit(1)

    print(f"\nPredicting {len(images)} image(s) with {len(predictors)} model(s):\n")
    rows = []  # (image, run_name, prediction)
    for img in images:
        print(f"{img}")
        for predictor in predictors:
            try:
                text = predictor.predict(img)
            except Exception as e:
                text = f"<error: {type(e).__name__}: {e}>"
            print(f"    {predictor.run_name:<32} -> {text}")
            rows.append((str(img), predictor.run_name, text))
        print()

    if args.save:
        import csv
        with open(args.save, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "model", "prediction"])
            writer.writerows(rows)
        print(f"Saved {len(rows)} rows to {args.save}")


if __name__ == "__main__":
    main()
