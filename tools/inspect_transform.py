"""
Visualize what the preprocessing pipeline does to a sample image.

Usage:
    python reah_ocr/tools/inspect_transform.py --config reah_ocr/config.yml
    python reah_ocr/tools/inspect_transform.py --config reah_ocr/config.yml --image path/to/image.png
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils import load_config
from architectures import build_transform, REGISTRY


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--image", default=None, help="Path to a specific image (optional)")
    parser.add_argument("--out", default="transform_preview.png", help="Output file path")
    return parser.parse_args()


def pick_sample_image(config):
    base = config["dataset"]["path"]
    label_path = os.path.join(base, config["dataset"]["train"]["labels"])
    image_dir = os.path.join(base, config["dataset"]["train"]["images"])
    with open(label_path, encoding="utf-8") as f:
        first_line = f.readline().strip()
    image_name = first_line.split("\t")[0]
    return os.path.join(image_dir, image_name)


def tensor_to_pil(tensor):
    arr = tensor.numpy()
    if arr.shape[0] == 1:
        arr = arr[0]
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def main():
    args = parse_args()
    config = load_config(args.config)

    arch_name = config["model"]["architecture"]
    transform = build_transform(config)

    image_path = args.image or pick_sample_image(config)
    print(f"Image      : {image_path}")
    print(f"Architecture: {arch_name}")
    print(f"Transform  :\n{transform}\n")

    original = Image.open(image_path).convert("L")
    processed_tensor = transform(original)
    processed_pil = tensor_to_pil(processed_tensor)

    h = max(original.size[1], processed_pil.size[1])
    gap = 10
    canvas = Image.new("L", (original.size[0] + gap + processed_pil.size[0], h), color=128)
    canvas.paste(original, (0, 0))
    canvas.paste(processed_pil, (original.size[0] + gap, 0))

    canvas.save(args.out)
    print(f"Saved preview: {args.out}")
    print("Left = original   |   Right = after transform")


if __name__ == "__main__":
    main()
