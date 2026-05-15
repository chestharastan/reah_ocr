import argparse
import os
import sys

import torch
from PIL import Image

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(1, os.path.abspath(os.path.join(_HERE, "..", "..", "traditional", "src")))

from vocab import KhmerVocab
from seq2seq import ChunkedSeq2Seq
from architectures.chunked_paper_seq2seq import _PaperChunkCNN

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def load_model(model_path, vocab, device):
    state = torch.load(model_path, map_location=device, weights_only=False)

    # Full model saved with torch.save(model, path)
    if hasattr(state, "eval"):
        print("Loaded full model object")
        state.to(device)
        state.eval()
        return state

    # Extract state dict from wrapper dicts
    if isinstance(state, dict):
        for key in ("model_state_dict", "model_state", "state_dict", "model"):
            if key in state:
                print(f"Checkpoint keys: {list(state.keys())}")
                if "epoch" in state:
                    print(f"Epoch: {state.get('epoch')}, val_cer: {state.get('val_cer')}")
                state = state[key]
                break

    # Infer num_classes from classifier weight
    num_classes = None
    for k, v in state.items():
        if "classifier.weight" in k:
            num_classes = v.shape[0]
            print(f"Inferred num_classes={num_classes} from {k}")
            break

    if num_classes is None:
        print("ERROR: could not infer num_classes from checkpoint.")
        sys.exit(1)

    if num_classes != len(vocab):
        print(f"WARNING: checkpoint has {num_classes} classes but charset gives vocab size {len(vocab)}.")
        print(f"         You are probably using the wrong charset.json for this checkpoint.")
        print(f"         Find the charset that was used when training this model.")

    cnn = _PaperChunkCNN(in_channels=1, out_channels=256)
    model = ChunkedSeq2Seq(
        cnn=cnn,
        cnn_out_channels=256,
        num_classes=num_classes,
        pad_id=vocab.pad_id,
        bos_id=vocab.bos_id,
        eos_id=vocab.eos_id,
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        chunk_width=100,
        overlap=16,
        max_target_len=256,
    )

    # Initialize LazyLinear before loading weights
    with torch.no_grad():
        dummy_img = torch.zeros(1, 1, 48, 512)
        dummy_tgt = torch.full((1, 1), vocab.bos_id, dtype=torch.long)
        model(dummy_img, dummy_tgt)

    result = model.load_state_dict(state, strict=True)
    if result.missing_keys:
        print(f"WARNING: missing keys in checkpoint: {result.missing_keys}")
    if result.unexpected_keys:
        print(f"WARNING: unexpected keys in checkpoint: {result.unexpected_keys}")
    model.to(device)
    model.eval()
    return model


def build_transform():
    from torchvision import transforms
    # No Grayscale() here — match training exactly: dataset.py loads as "L" mode
    # then applies Resize+ToTensor+Normalize with no grayscale conversion step.
    return transforms.Compose([
        transforms.Resize((48, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


def collect_images(paths, recursive=False):
    image_paths = []
    for entry in paths:
        if os.path.isdir(entry):
            if recursive:
                for root, _, files in os.walk(entry):
                    for fname in sorted(files):
                        if fname.lower().endswith(IMAGE_EXTS):
                            image_paths.append(os.path.join(root, fname))
            else:
                for fname in sorted(os.listdir(entry)):
                    if fname.lower().endswith(IMAGE_EXTS):
                        image_paths.append(os.path.join(entry, fname))
        else:
            image_paths.append(entry)
    return image_paths


def predict(image_path, model, transform, vocab, device, debug=False):
    img = Image.open(image_path).convert("L")  # match training: dataset.py loads as "L"
    if debug:
        print(f"    Original size: {img.size[0]}w x {img.size[1]}h")
    tensor = transform(img).unsqueeze(0).to(device)

    if debug:
        import torch.nn.functional as F
        with torch.no_grad():
            memory = model.encode(tensor)
            print(f"    Memory shape: {tuple(memory.shape)}")
            print(f"    Memory stats: mean={memory.mean().item():.4f}  std={memory.std().item():.4f}  "
                  f"min={memory.min().item():.4f}  max={memory.max().item():.4f}")
            # first-token logits
            bos = torch.full((1, 1), model.bos_id, dtype=torch.long, device=device)
            logits = model.decode_step(memory, bos)  # (1, 1, vocab)
            probs = F.softmax(logits[0, 0], dim=-1)
            top5 = probs.topk(5)
            top5_ids = top5.indices.tolist()
            top5_probs = top5.values.tolist()
            decoded = [vocab.idx_to_char[i] if i < len(vocab.idx_to_char) else f"<{i}>" for i in top5_ids]
            print(f"    Top-5 first tokens: {list(zip(top5_ids, decoded, [f'{p:.3f}' for p in top5_probs]))}")

    token_ids = model.generate(tensor)
    ids = token_ids[0].tolist()
    if debug:
        print(f"    Raw token ids: {ids[:20]}{'...' if len(ids) > 20 else ''}")
    return vocab.decode(ids)


def main():
    parser = argparse.ArgumentParser(description="Khmer OCR inference (chunkmerge seq2seq)")
    parser.add_argument("images", nargs="+", help="Image path(s) or folder(s) to predict")
    parser.add_argument("--model", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--charset", required=True, help="Path to charset.json")
    parser.add_argument("--recursive", "-r", action="store_true",
                        help="Scan folders recursively for images")
    parser.add_argument("--debug", action="store_true",
                        help="Print image size and raw token ids for each prediction")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    vocab = KhmerVocab(charset_path=args.charset)
    print(f"Vocab size: {len(vocab)}")

    model = load_model(args.model, vocab, device)
    transform = build_transform()

    image_paths = collect_images(args.images, recursive=args.recursive)

    if not image_paths:
        print("No images found.")
        sys.exit(0)

    print(f"\nPredicting {len(image_paths)} image(s):\n")
    for img_path in image_paths:
        try:
            result = predict(img_path, model, transform, vocab, device, debug=args.debug)
            print(f"  {img_path}")
            print(f"  → {result if result else '(empty — check charset match or image format)'}")
            print()
        except Exception as e:
            print(f"  {img_path}")
            print(f"  ERROR: {e}")
            print()


if __name__ == "__main__":
    main()
