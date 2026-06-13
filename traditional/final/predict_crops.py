"""Adapter predictor for the Detection-system OCR router.

The Detection app crops annotation boxes into a temp folder and shells out to a
predictor, then parses stdout looking for, per crop:

    <path ending in box_XXXX.png>
    → <predicted text>          (or)   ERROR: <message>

The repo's own ``final/predict.py`` already knows how to rebuild *every*
architecture (including ``vgg_bilstm_ctc``) from its saved ``experiment.json``.
This wrapper reuses that machinery for a single run and emits exactly the line
format the router parses, so the VGG model becomes selectable in the app
without touching the proven inference code.

Usage:
    python final/predict_crops.py <folder-or-images> --run <run_name> \
        [--outputs-root DIR] [--charset PATH] [--device cuda|cpu]
"""

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Reuse the discovery + Predictor machinery from the sibling script.
from predict import (  # noqa: E402
    DEFAULT_CHARSET,
    DEFAULT_OUTPUTS_ROOT,
    gather_images,
    load_predictors,
)


def main():
    parser = argparse.ArgumentParser(
        description="OCR a folder of crops with one trained run (router adapter)."
    )
    parser.add_argument("images", nargs="+", help="Image file(s) and/or folder(s)")
    parser.add_argument("--run", required=True, help="Run name to use (e.g. vgg_bilstm_ctc_10k)")
    parser.add_argument("--outputs-root", default=str(DEFAULT_OUTPUTS_ROOT))
    parser.add_argument("--charset", default=str(DEFAULT_CHARSET))
    parser.add_argument("--device", default=None, help="cuda / cpu (auto by default)")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    predictors = load_predictors(args.outputs_root, args.charset, device, only={args.run})
    if not predictors:
        print(f"ERROR: run '{args.run}' not found under {args.outputs_root}", file=sys.stderr)
        sys.exit(1)
    predictor = predictors[0]

    for img in gather_images(args.images):
        print(img)
        try:
            print(f"→ {predictor.predict(img)}")
        except Exception as exc:  # one bad crop must not sink the batch
            print(f"ERROR: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
