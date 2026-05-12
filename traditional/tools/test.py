import argparse
import os
import sys

import torch
from PIL import Image
import torchvision.transforms as T

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vocab import KhmerVocab


def load_vocab(charset_path):
    vocab = KhmerVocab(charset_path)
    print(f"Charset: {len(vocab)} classes (including blank)")
    return vocab


def load_model(model_path, num_classes, arch, device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Full model saved with torch.save(model, path)
    if hasattr(checkpoint, "eval"):
        print("Loaded full model")
        checkpoint.to(device)
        checkpoint.eval()
        return checkpoint

    if not isinstance(checkpoint, dict):
        print(f"ERROR: unexpected checkpoint type: {type(checkpoint)}")
        sys.exit(1)

    print(f"Checkpoint keys: {list(checkpoint.keys())}")

    # Extract state dict from checkpoint wrapper
    if "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
        print(f"Epoch: {checkpoint.get('epoch')}, val_cer: {checkpoint.get('val_cer')}")
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        # Assume the dict itself is a raw state dict
        state_dict = checkpoint

    # Infer num_classes from classifier weight if not provided
    if num_classes is None:
        for key, tensor in state_dict.items():
            if "classifier" in key and "weight" in key:
                num_classes = tensor.shape[0]
                print(f"Inferred num_classes={num_classes} from {key}")
                break

    if num_classes is None:
        print("ERROR: could not infer num_classes from checkpoint. Pass --num-classes explicitly.")
        sys.exit(1)

    # Build model
    if arch == "cnn_bilstm_ctc_skel":
        from architectures.cnn_bilstm_ctc_skel import Model
    elif arch == "cnn_bilstm_ctc":
        from architectures.cnn_bilstm_ctc import Model
    else:
        print(f"ERROR: unknown architecture '{arch}'. Use cnn_bilstm_ctc or cnn_bilstm_ctc_skel.")
        sys.exit(1)

    model = Model(num_classes=num_classes)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Built {arch} with num_classes={num_classes}")
    return model


def build_transform():
    return T.Compose([
        T.Grayscale(),
        T.Resize((64, 512)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])


def predict(image_path, model, transform, vocab, device):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)          # (1, time, num_classes)
        output = output.squeeze(0)   # (time, num_classes)
        preds = torch.argmax(output, dim=-1)  # (time,)

    return vocab.decode(preds.tolist())


def main():
    parser = argparse.ArgumentParser(description="Khmer OCR Inference")
    parser.add_argument("images", nargs="+", help="Image path(s) to predict")
    parser.add_argument("--model", default="outputs/khmer_ocr_100k_v1/checkpoints/best_model.pth",
                        help="Path to saved model checkpoint")
    parser.add_argument("--charset", default="ocr_data_100k/charset.json",
                        help="Path to charset.json")
    parser.add_argument("--arch", default="cnn_bilstm_ctc_skel",
                        choices=["cnn_bilstm_ctc", "cnn_bilstm_ctc_skel"],
                        help="Model architecture")
    parser.add_argument("--num-classes", type=int, default=None,
                        help="Override num_classes (auto-detected from checkpoint by default)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    vocab = load_vocab(args.charset)
    model = load_model(args.model, args.num_classes, args.arch, device)
    transform = build_transform()

    print(f"\nPredicting {len(args.images)} image(s):\n")
    for img_path in args.images:
        try:
            result = predict(img_path, model, transform, vocab, device)
            print(f"  {img_path}")
            print(f"  → {result}")
            print()
        except Exception as e:
            print(f"  {img_path}")
            print(f"  ERROR: {e}")
            print()


if __name__ == "__main__":
    main()
