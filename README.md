# reah_ocr — Khmer OCR

A Khmer Optical Character Recognition system using CNN + BiLSTM + CTC loss. Built specifically for the 3-zone vertical structure of Khmer script (above-base vowels, base consonants, subscript/coeng).

---

## Project Structure

```
reah_ocr/
├── config.yml                      # Central configuration
├── docs/
│   └── codebase.md                 # Technical documentation
├── src/
│   ├── architectures/
│   │   ├── __init__.py             # Auto-discovery registry for models
│   │   └── cnn_bilstm_ctc.py       # CNN+BiLSTM+CTC model
│   ├── collate.py                  # DataLoader collate function
│   ├── dataset.py                  # PyTorch Dataset
│   ├── preprocessing.py            # Standard image transform pipeline
│   ├── processing.py               # Khmer-specific preprocessing (Stage 1)
│   ├── train_loop.py               # Single-epoch training function
│   ├── transforms.py               # Optional transforms (binary, skeleton)
│   ├── utils.py                    # Config loader, checkpoint save/load
│   ├── validate.py                 # Validation loop + CER metric
│   └── vocab.py                    # Khmer vocabulary encoder/decoder
└── tools/
    └── train.py                    # Main training entry point
```

---

## Requirements

- Python 3.8+
- PyTorch + torchvision
- OpenCV
- scikit-image
- PyYAML
- Pillow
- NumPy
- tqdm

Install all dependencies:

```bash
pip install torch torchvision opencv-python scikit-image pyyaml pillow numpy tqdm
```

---

## Setup

### 1. Dataset Format

Organize your data as:

```
dataset/
├── train/
│   ├── images/
│   └── labels.txt
└── val/
    ├── images/
    └── labels.txt
```

Each line in `labels.txt`:
```
image_filename.png\tkhmer_text
```

You also need a `charset.json` file listing all Khmer characters used.

### 2. Configuration

Edit `config.yml` to point to your dataset and tune hyperparameters:

```yaml
dataset:
  charset: path/to/charset.json
  path: path/to/dataset/
  train:
    images: train/images
    labels: train/labels.txt
  val:
    images: val/images
    labels: val/labels.txt

preprocessing:
  image_height: 64
  image_width: 512

model:
  architecture: cnn_bilstm_ctc

training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001

checkpoint:
  checkpoint_dir: checkpoints/
  save_every: 10
```

---

## Usage

### Preprocess Images (Optional)

Run the Khmer-specific Stage 1 preprocessing on a single image:

```bash
python src/processing.py path/to/image.png
# Outputs saved to ./output/stage1_*.png
```

This applies zone-aware CLAHE, Sauvola adaptive binarization, and noise cleanup tuned for Khmer script.

### Train

```bash
python tools/train.py
```

With a custom config:

```bash
python tools/train.py --config path/to/custom_config.yml
```

Resume from a checkpoint:

```bash
python tools/train.py --resume path/to/checkpoint.pth
```

### Checkpoints

| File | Contents |
|------|----------|
| `best_model.pth` | Best validation CER (weights only) |
| `last_model.pth` | Latest epoch (model + optimizer, resumable) |
| `epoch_NNN.pth` | Periodic checkpoint every N epochs |

---

## Model Architecture

```
Input Image (1 × 64 × 512)
        ↓
CNN Backbone (3 conv layers: 64 → 128 → 256 channels)
        ↓
Feature Reshape → Sequence
        ↓
BiLSTM (2 layers, hidden_size=256, bidirectional)
        ↓
Linear (512 → num_classes)
        ↓
CTC Loss / Greedy Decode
```

---

## Key Design Details

- **CTC Loss** — handles variable-length label sequences without forced alignment
- **Greedy decoding** — removes blanks and consecutive duplicates at inference
- **Validation metric** — Character Error Rate (CER) via Levenshtein distance
- **Gradient clipping** — `max_norm=5.0` for stable training
- **LR scheduler** — `ReduceLROnPlateau` on validation CER
- **Architecture registry** — drop a new `.py` file in `src/architectures/` to auto-register it
- **Device agnostic** — automatically uses CUDA if available
