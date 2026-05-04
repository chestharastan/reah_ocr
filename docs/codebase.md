# Codebase Reference

## Project Structure

```
reah_ocr/
├── main.py
├── src/
│   ├── model.py
│   ├── preprocessing.py
│   ├── processing.py
│   ├── dataset.py
│   ├── collate.py
│   ├── vocab.py
│   ├── train_loop.py
│   └── utils.py
├── tools/
│   └── train.py
├── setting/
│   └── config.yml
└── docs/
    └── codebase.md
```

---

## src/model.py

Defines the `OCRModel` neural network — a CNN + BiLSTM + CTC architecture.

**Flow:**
1. **CNN** — 3 convolutional layers extract visual features from the grayscale image
2. **Reshape** — feature map is rearranged from `(batch, channels, height, width)` into a time-step sequence `(batch, width, channels*height)` so the RNN can process it left-to-right
3. **BiLSTM** — 2-layer bidirectional LSTM reads the sequence in both directions to capture context
4. **Classifier** — a linear layer maps each time step to a character score over the full vocabulary

**Input:** grayscale image tensor `(batch, 1, H, W)`  
**Output:** raw logits `(batch, time_steps, num_classes)` — fed into CTC loss during training

---

## src/preprocessing.py

Defines `get_transform(image_height, image_width)` — returns a torchvision transform pipeline used at both training and inference time.

**Steps applied to each image:**
1. Resize to the target `(image_height, image_width)` from config
2. Convert PIL image to a PyTorch tensor
3. Normalize pixel values: mean `0.5`, std `0.5` — maps `[0, 1]` to `[-1, 1]`

---

## src/processing.py

Stage 1 image preprocessing pipeline built specifically for Khmer script. This runs **before** the image enters the model.

**Why it exists:** Khmer has 3 vertical zones (above-base vowels, base consonant, subscript/coeng). Standard binarization destroys thin diacritics. This pipeline handles that.

**Steps:**
1. **Grayscale conversion** — handles BGR, RGBA, and already-gray inputs
2. **Zone-aware CLAHE** — applies contrast normalization independently to each of the 3 zones on the grayscale image, before binarization, so thin diacritics are not crushed by thick strokes
3. **Sauvola binarization** — adaptive local thresholding that handles uneven ink pressure (much better than Otsu for handwritten Khmer)
4. **Zone detection** — analyzes the horizontal ink density profile to find the 3 zone boundaries automatically, no Unicode metadata needed
5. **Noise cleanup** — removes only fully isolated single-pixel specks; does not erode thin strokes
6. **Zone masks + debug overlay** — produces boolean masks per zone and a color-annotated BGR image for visualization

**Main entry point:** `preprocess(image)` returns a `PreprocessingResult` dataclass with `binary`, `zones`, `zone_a_mask`, `zone_b_mask`, `zone_c_mask`, `debug_overlay`.

Can also be run from the command line:
```bash
python processing.py <image_path>
# saves output to ./output/stage1_*.png
```

---

## src/dataset.py

`OCRDataset` — PyTorch `Dataset` that loads image-label pairs from disk.

**Label file format:** a tab-separated `.txt` file where each line is:
```
image_filename.png\ttext label
```

**What it does per sample:**
- Opens the image file as grayscale (`PIL` `"L"` mode)
- Applies the transform (resize + normalize)
- Returns `(image_tensor, text_string)`

---

## src/collate.py

`ocr_collate_fn(batch, vocab)` — custom collate function passed to `DataLoader`.

The default PyTorch collate cannot handle variable-length label strings, so this handles batching manually:

- Stacks all image tensors into one batch tensor
- Encodes each text label into integer indices using `vocab.encode()`
- Concatenates all label indices into a flat tensor (required by CTC loss)
- Records the length of each label in a separate tensor

**Returns:** `(images, labels, label_lengths)` — exactly the format `nn.CTCLoss` expects.

---

## src/vocab.py

`KhmerVocab` — defines the full character set used for Khmer OCR.

**Character set includes:**
- All 33 Khmer base consonants
- All main vowel signs (above, below, pre, post)
- Diacritics: anusvara, visarga, etc.
- Combining marks: coeng `្`, etc.
- Space character

**Index 0** is always the CTC blank token `<blank>`.

**Key methods:**
- `encode(text)` — converts a string to a list of integer indices; unknown characters are skipped with a warning
- `decode(indices)` — converts model output indices back to a string, applying CTC collapsing (removes consecutive duplicates and blank tokens)
- `__len__()` — returns total vocabulary size including blank

---

## src/train_loop.py

`train_one_epoch(model, dataloader, optimizer, criterion, device)` — runs one complete pass over the training data.

**Per batch:**
1. Move images, labels, and label lengths to the target device
2. Forward pass through the model
3. Permute output to `(time, batch, classes)` — required by CTC loss
4. Apply `log_softmax`
5. Compute `input_lengths` (same sequence length for every item in the batch)
6. Compute CTC loss
7. Backpropagate and update weights

**Returns:** average loss over all batches in the epoch.

---

## src/utils.py

Three utility functions used by the training script.

| Function | What it does |
|---|---|
| `load_config(path)` | Reads a YAML file and returns it as a Python dict |
| `save_checkpoint(model, optimizer, epoch, val_cer, checkpoint_dir)` | Saves model + optimizer state as `epoch_NNN.pth` and also overwrites `last_model.pth` |
| `save_best_model(model, checkpoint_dir)` | Saves only the model weights (no optimizer) as `best_model.pth` |

Checkpoint directory is created automatically if it does not exist.

---

## tools/train.py

The main training script. Run this to start training.

**What it does end-to-end:**
1. Loads `setting/config.yml`
2. Selects CUDA or CPU based on availability and config
3. Builds the image transform pipeline
4. Creates `OCRDataset` for training data (path resolved from `dataset.path` + `dataset.train.*`)
5. Builds `KhmerVocab` and wraps `ocr_collate_fn` with it using `functools.partial`
6. Creates `DataLoader` with shuffling
7. Instantiates `OCRModel` and moves it to device
8. Sets up `CTCLoss` (blank index 0) and `Adam` optimizer
9. Runs the training loop for `N` epochs — each epoch calls `train_one_epoch()`, prints loss, saves a checkpoint, and saves `best_model.pth` if validation CER improved

**How to run:**
```bash
cd reah_ocr
python tools/train.py
```

---

## setting/config.yml

Central configuration file. All paths, hyperparameters, and settings live here so nothing is hardcoded in the scripts.

| Section | Controls |
|---|---|
| `project` | Project name and output directory |
| `dataset` | Base dataset path and train/val subfolder names |
| `preprocessing` | Image resize dimensions and normalization flags |
| `model` | Architecture, hidden size, layers, dropout |
| `training` | Epochs, batch size, learning rate, optimizer, device |
| `checkpoint` | Save frequency, best-model tracking, checkpoint directory, resume path |
