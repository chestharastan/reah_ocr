"""Adapter predictor: OCR a folder of crops with ONE model directory.

``predict_crops.py`` discovers runs by the ``<run>/checkpoints/`` layout. The
``Khmer-OCR/Model/final/*`` runs instead keep ``best_model.pth`` +
``experiment.json`` directly inside the run directory, so this adapter points
straight at one such directory and emits exactly the line format the Detection
app's OCR router parses, per crop:

    <path ending in box_XXXX.png>
    → <predicted text>            (or)   ERROR: <message>

The charset must be passed explicitly because the path baked into
``experiment.json`` is the training machine's path (does not exist here).

Usage:
    python final/predict_model_dir.py <folder-or-images> --model-dir DIR \
        --charset PATH [--device cuda|cpu]
"""

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Reuse the proven Predictor + image-gathering machinery.
from predict import DEFAULT_CHARSET, Predictor, gather_images  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="OCR a folder of crops with one model directory (router adapter)."
    )
    parser.add_argument("images", nargs="+", help="Image file(s) and/or folder(s)")
    parser.add_argument(
        "--model-dir", required=True,
        help="Directory holding best_model.pth (or last_model.pth) + experiment.json",
    )
    parser.add_argument("--charset", default=str(DEFAULT_CHARSET))
    parser.add_argument("--device", default=None, help="cuda / cpu (auto by default)")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_dir = Path(args.model_dir)

    checkpoint = model_dir / "best_model.pth"
    if not checkpoint.exists():
        checkpoint = model_dir / "last_model.pth"
    if not checkpoint.exists():
        print(f"ERROR: no best_model.pth/last_model.pth in {model_dir}", file=sys.stderr)
        sys.exit(1)

    with open(model_dir / "experiment.json") as f:
        config = json.load(f)["config"]

    # Use the charset passed by the caller (the config's path is the training
    # machine's and is not present here).
    predictor = Predictor(model_dir.name, config, checkpoint, args.charset, device)

    for img in gather_images(args.images):
        print(img)
        try:
            print(f"→ {predictor.predict(img)}")
        except Exception as exc:  # one bad crop must not sink the batch
            print(f"ERROR: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
