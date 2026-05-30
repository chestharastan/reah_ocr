# Training Process Flow — Khmer OCR

> This file explains the full pipeline from one image to the final trained model.
> Use this as a reference for building your thesis slide process flow diagram.

---

## Overview: 4 Model Variants

Your project trains 4 main variants (2 backbones × 2 preprocessing types):

| Name    | Backbone       | Preprocessing |
|---------|---------------|---------------|
| **CBC** | Custom CNN     | Normal        |
| **CBC-ZS** | Custom CNN  | + Skeleton    |
| **RBC** | ResNet         | Normal        |
| **RBC-ZS** | ResNet      | + Skeleton    |

Each variant is trained on 3 dataset sizes: **10k, 50k, 100k** images.

---

## Full Process Flow (Step by Step)

```
[Input Image]
      ↓
[Step 1: Load Image + Label]
      ↓
[Step 2: Preprocessing]
      ↓ (Normal path)          ↓ (Skeleton path)
   Resize → Grayscale      Resize → Grayscale → Skeletonize
   → Normalize              → ToTensor
   → ToTensor
      ↓                         ↓
[Step 3: Batch Chunking — DataLoader groups images into batches]
      ↓
[Step 4: Encode Labels (Vocabulary)]
      ↓
[Step 5: CNN Backbone — Extract visual features]
      ↓
[Step 6: BiLSTM — Read features as a sequence (left ↔ right)]
      ↓
[Step 7: Linear Classifier — Predict character at each time step]
      ↓
[Step 8: CTC Loss — Compare prediction vs. true label]
      ↓
[Step 9: Backpropagation — Update model weights]
      ↓
[Step 10: Validation — Measure CER (Character Error Rate)]
      ↓
[Step 11: Save Checkpoint — Save last + best model]
      ↓
[Output: best_model.pth — Trained model weights]
```

---

## Step-by-Step Detail

### Step 1 — Load Image + Label
- **File:** `src/dataset.py` → `OCRDataset`
- Reads `labels.txt` file, which maps `image_filename → Khmer text`
- Opens each image as **grayscale** (mode `"L"`)
- One sample = (image, text string)

**Slide bullet:**
> Load image file and its corresponding Khmer text label from `labels.txt`

---

### Step 2 — Preprocessing (DIFFERENT for Normal vs Skeleton)

#### Normal Preprocessing
- **File:** `src/architectures/resnet_bilstm_ctc.py` → `get_transform()`
- Steps applied to every image:
  1. **Resize** → fixed size 64 × 512 pixels (height × width)
  2. **ToTensor** → convert PIL image to PyTorch tensor (values 0.0–1.0)
  3. **Normalize** → shift pixel values: mean=0.5, std=0.5 → range (−1 to +1)

**Math — Normalization:**

$$\hat{x} = \frac{x - \mu}{\sigma} = \frac{x - 0.5}{0.5}$$

where $x \in [0, 1]$ is the raw pixel value after ToTensor, $\mu = 0.5$, $\sigma = 0.5$.
Result: $\hat{x} \in [-1, +1]$.

**Slide bullet (Normal):**
> Resize image to 64×512 → convert to tensor → normalize pixel values to [−1, +1]

---

#### Skeleton Preprocessing
- **File:** `src/architectures/resnet_bilstm_ctc_skel.py` → `SkeletonTransform`
- Same resize, but adds a **skeletonization** step between resize and ToTensor:
  1. **Resize** → 64 × 512 pixels
  2. **Skeletonize** → morphological thinning (OpenCV):
     - Threshold image to black/white binary
     - Repeatedly erode + detect boundary pixels
     - Reduce every stroke to **1-pixel-wide centerlines**
     - Loop until no ink pixels remain
  3. **ToTensor** → convert to tensor

**Math — Skeletonization (Morphological Thinning):**

Each iteration of the loop computes:

$$E_k = B_{k-1} \ominus S$$

$$D_k = E_k \oplus S$$

$$\text{Skel} = \text{Skel} \cup (B_{k-1} \setminus D_k)$$

$$B_k = E_k$$

where:
- $B_k$ = binary image at iteration $k$ (foreground pixels)
- $S$ = 3×3 cross-shaped structuring element
- $\ominus$ = morphological erosion (shrink foreground)
- $\oplus$ = morphological dilation (expand foreground)
- $\setminus$ = set difference (boundary pixels)

Loop stops when $B_k = \emptyset$ (no foreground pixels remain).
Each iteration peels one layer from every stroke until only the 1-pixel centerline survives.

**What skeletonization does:**
Removes stroke thickness — a thick stroke of 10 pixels becomes a single line 1 pixel wide.
This forces the model to learn **pure shape/structure** rather than stroke weight.

**Slide bullet (Skeleton):**
> Resize → Binarize → Thin all strokes to 1-pixel centerlines (morphological skeletonization) → ToTensor

---

### Step 3 — Batch Chunking (DataLoader)
- **File:** `tools/train.py` → `DataLoader`
- Groups individual images into **batches** (default: 64 images per batch)
- **Why batch?** Processing 64 images at once is much faster on GPU than 1 at a time
- Shuffles training data every epoch so the model doesn't memorize order
- Uses `collate_fn` to stack images and pad/encode labels into tensors

**Slide bullet:**
> Group 64 images into one batch → stack into a single tensor for GPU processing

---

### Step 4 — Encode Labels (Vocabulary)
- **File:** `src/vocab.py` → `KhmerVocab`
- Loads `charset.json` — the list of all Khmer characters the model can predict
- Converts each Khmer character to an integer ID
  - Example: `"ក"` → `1`, `"ខ"` → `2`, etc.
  - ID `0` is always reserved for the **blank** token (used by CTC)
- `collate.py` converts text labels → integer sequences for the whole batch

**Slide bullet:**
> Convert Khmer characters to integer IDs using a character dictionary (charset)

---

### Step 5 — CNN Backbone (Feature Extraction)
- **File:** `resnet_bilstm_ctc.py` or `cnn_bilstm_ctc.py`
- The CNN reads the image and produces a **feature map** — a compact representation
  of what visual patterns exist and where

#### Normal CNN (CBC) — 3 Conv blocks
```
Input (1 × 64 × 512) → Conv+BN+ReLU → MaxPool
                      → Conv+BN+ReLU → MaxPool
                      → Conv+BN+ReLU
Output: (256 × 16 × 128)
```

#### ResNet (RBC) — Residual blocks with skip connections
```
Input (1 × 64 × 512) → Stem Conv → MaxPool
                      → ResBlock Stage 1 (64ch, same size)
                      → ResBlock Stage 2 (128ch, halved)
                      → ResBlock Stage 3 (256ch, deepened)
Output: (256 × 16 × 128)
```
**Math — Convolution + Batch Norm + ReLU (one Conv block):**

$$z = W * x + b \quad \text{(convolution)}$$

$$\hat{z}_i = \frac{z_i - \mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}} \quad \text{(batch normalization)}$$

$$y_i = \gamma \hat{z}_i + \beta \quad \text{(learnable scale and shift)}$$

$$\text{output} = \max(0,\ y_i) \quad \text{(ReLU)}$$

where $\mu_\mathcal{B}$ and $\sigma_\mathcal{B}^2$ are the mean and variance of the current mini-batch.

**Math — ResNet Skip Connection:**

$$\text{ResBlock}(x) = \text{ReLU}\bigl(\mathcal{F}(x) + x\bigr)$$

where $\mathcal{F}(x)$ = two Conv+BN layers. The $+x$ term is the skip connection.
If channel sizes differ, a 1×1 convolution is applied to $x$ before adding:

$$\text{ResBlock}(x) = \text{ReLU}\bigl(\mathcal{F}(x) + W_s x\bigr)$$

**ResNet advantage:** Skip connections (`output = conv(x) + x`) prevent vanishing gradients — easier to train deeper networks.

**Math — Reshape (CNN → Sequence):**

Feature map of shape $(B, C, H, W)$ is permuted and reshaped to $(B, W, C \times H)$:

$$\mathbf{f}_t = \text{vec}(\text{features}[:, :, :, t]) \in \mathbb{R}^{C \times H}$$

so each time step $t \in \{1, \ldots, W\}$ is one vertical column of the feature map (dimension = $256 \times 16 = 4096$).

- After CNN, the feature map is **reshaped** from 2D (height × width) into a **1D sequence** of time steps
  - Shape becomes: (batch, 128 time steps, 4096 features per step)

**Slide bullet:**
> CNN extracts visual features from the image → reshape into a sequence of 128 time steps

---

### Step 6 — BiLSTM (Sequence Modeling)
- **File:** same architecture files → `self.rnn`
- **BiLSTM** = Bidirectional Long Short-Term Memory
- Reads the 128-step sequence **left-to-right AND right-to-left** simultaneously
- Combines both directions — each time step can see context from both sides
- Parameters: hidden_size=256, 2 layers, dropout=0.3

**Math — LSTM Cell (one direction):**

At each time step $t$, given input $x_t$ and previous hidden state $h_{t-1}$:

$$f_t = \sigma(W_f [h_{t-1},\, x_t] + b_f) \quad \text{(forget gate)}$$

$$i_t = \sigma(W_i [h_{t-1},\, x_t] + b_i) \quad \text{(input gate)}$$

$$\tilde{c}_t = \tanh(W_c [h_{t-1},\, x_t] + b_c) \quad \text{(candidate cell)}$$

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \quad \text{(cell state update)}$$

$$o_t = \sigma(W_o [h_{t-1},\, x_t] + b_o) \quad \text{(output gate)}$$

$$h_t = o_t \odot \tanh(c_t) \quad \text{(hidden state output)}$$

where $\sigma$ = sigmoid, $\odot$ = element-wise multiplication.

**Math — Bidirectional Combination:**

$$\overrightarrow{h}_t = \text{LSTM}_{\text{fwd}}(x_t,\, \overrightarrow{h}_{t-1}) \quad \text{(left → right)}$$

$$\overleftarrow{h}_t = \text{LSTM}_{\text{bwd}}(x_t,\, \overleftarrow{h}_{t+1}) \quad \text{(right → left)}$$

$$h_t = \bigl[\overrightarrow{h}_t \,;\, \overleftarrow{h}_t\bigr] \in \mathbb{R}^{2 \times \text{hidden\_size}} = \mathbb{R}^{512}$$

**Why bidirectional?**
A character like a vowel in Khmer depends on both the consonant before AND after it.
Reading both directions helps the model understand this context.

**Slide bullet:**
> BiLSTM reads the sequence forward and backward → captures context from both directions

---

### Step 7 — Linear Classifier
- **File:** same architecture files → `self.classifier`
- A simple `Linear` layer applied at every time step
- Maps the BiLSTM output (512-dim) → probability over all characters in vocabulary
- Output shape: (batch, 128 time steps, vocab_size)
- Each time step predicts: "what character (or blank) is most likely here?"

**Math — Linear Layer + Softmax:**

$$y_t = W \cdot h_t + b \in \mathbb{R}^{|\mathcal{V}|}$$

$$P(c \mid t) = \text{softmax}(y_t)_c = \frac{e^{y_{t,c}}}{\displaystyle\sum_{k=1}^{|\mathcal{V}|} e^{y_{t,k}}}$$

where $|\mathcal{V}|$ = vocabulary size (number of Khmer characters + blank token), and $P(c \mid t)$ is the probability of character $c$ at time step $t$.

**Slide bullet:**
> Linear layer converts each time step into character probabilities over the vocabulary

---

### Step 8 — CTC Loss
- **File:** `tools/train.py` → `nn.CTCLoss`
- **CTC = Connectionist Temporal Classification**
- The model outputs 128 predictions but the text might only be 5 characters long
- CTC handles this: it considers ALL possible alignments between output and label
- The blank token (ID=0) acts as a separator
- Example: `[blank, ក, ក, blank, ខ, blank]` → decoded as `"កខ"` (duplicate removal)
- CTC loss measures how wrong the prediction is compared to the true label

**Math — CTC Loss:**

$$\mathcal{L}_{\text{CTC}} = -\log P(\mathbf{l} \mid \mathbf{x})$$

$$P(\mathbf{l} \mid \mathbf{x}) = \sum_{\pi \,\in\, \mathcal{B}^{-1}(\mathbf{l})} \prod_{t=1}^{T} P(\pi_t \mid t)$$

where:
- $\mathbf{l}$ = the true label sequence (e.g., "កខ")
- $\mathbf{x}$ = the input image
- $T$ = number of time steps (128)
- $\pi$ = one possible alignment path (sequence of characters + blanks of length $T$)
- $\mathcal{B}^{-1}(\mathbf{l})$ = all valid paths that collapse to $\mathbf{l}$ after removing blanks and duplicate consecutive characters
- $P(\pi_t \mid t)$ = probability of outputting symbol $\pi_t$ at time $t$ (from Softmax)

The CTC collapsing function $\mathcal{B}$ removes blanks and duplicates:

$$\mathcal{B}(\text{[blank, ក, ក, blank, ខ, blank]}) = \text{"កខ"}$$

**Slide bullet:**
> CTC Loss compares 128 time-step predictions to the true label — handles variable-length alignment automatically

---

### Step 9 — Backpropagation (Learning)
- **File:** `src/train_loop.py`
- After computing the loss:
  1. `loss.backward()` — compute how much each weight contributed to the error
  2. **Gradient clipping** (`max_norm=5.0`) — cap extreme gradients to prevent exploding updates
  3. `optimizer.step()` — update all weights using **Adam optimizer** (lr=0.001)
- The **learning rate scheduler** (`ReduceLROnPlateau`) automatically reduces the learning rate when validation performance stops improving (patience=5 epochs, factor=0.5)

**Math — Gradient Clipping:**

$$g \leftarrow g \cdot \min\!\left(1,\ \frac{\tau}{\|g\|_2}\right), \quad \tau = 5.0$$

Scales down the gradient vector if its norm exceeds threshold $\tau$, preventing exploding updates.

**Math — Adam Optimizer:**

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1)\, g_t \quad \text{(1st moment — mean)}$$

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2)\, g_t^2 \quad \text{(2nd moment — variance)}$$

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \quad \text{(bias correction)}$$

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon}\, \hat{m}_t$$

Defaults used: $\eta = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$.

**Math — Learning Rate Scheduler:**

$$\eta_{\text{new}} = \eta \times 0.5 \quad \text{if val\_CER does not improve for 5 consecutive epochs}$$

$$\eta_{\min} = 10^{-6} \quad \text{(lower bound)}$$

**Slide bullet:**
> Compute gradients → clip → Adam optimizer updates all weights → learning rate reduces when validation stalls

---

### Step 10 — Validation (Measure CER)
- **File:** `src/validate.py`
- After every epoch, run on the **validation set** (data the model has NOT seen)
- Decode predictions using **Greedy CTC decoding**:
  - Take the most likely character at each time step
  - Remove blanks and duplicate consecutive characters
- Compute **CER (Character Error Rate)**:
  - CER = (edit distance between predicted text and true text) / (length of true text)
  - Edit distance = minimum insertions + deletions + substitutions needed
  - CER = 0.0 means perfect, CER = 1.0 means completely wrong

**Math — Greedy CTC Decoding:**

$$\hat{\pi}_t = \arg\max_c\, P(c \mid t) \quad \text{(most likely character at each step)}$$

$$\hat{\mathbf{l}} = \mathcal{B}(\hat{\pi}_1, \hat{\pi}_2, \ldots, \hat{\pi}_T) \quad \text{(collapse blanks and duplicates)}$$

**Math — Levenshtein Edit Distance:**

$$D(i, j) = \begin{cases}
i & \text{if } j = 0 \\
j & \text{if } i = 0 \\
D(i-1,\, j-1) & \text{if } s_i = \hat{s}_j \\
1 + \min\!\begin{cases} D(i-1,\, j) \\ D(i,\, j-1) \\ D(i-1,\, j-1) \end{cases} & \text{otherwise}
\end{cases}$$

where $s$ = true text, $\hat{s}$ = predicted text, operations are delete / insert / substitute.

**Math — CER:**

$$\text{CER} = \frac{\displaystyle\sum_{\text{samples}} D(s,\, \hat{s})}{\displaystyle\sum_{\text{samples}} |s|}$$

where $|s|$ = number of characters in the true text. Lower is better; 0.0 = perfect.

**Slide bullet:**
> Run model on validation set → decode predictions → compute CER (Character Error Rate)

---

### Step 11 — Save Checkpoint
- **File:** `tools/train.py` + `src/utils.py`
- Every epoch saves `last_model.pth` (so training can be resumed if interrupted)
- Every 5 epochs saves `epoch_005.pth`, `epoch_010.pth`, etc.
- If this epoch has the **best CER so far** → overwrites `best_model.pth`
- Also logs metrics to `metrics.csv` (epoch, train_loss, val_cer, lr, time)

**Slide bullet:**
> Save current weights as checkpoint → if best CER ever → save as best_model.pth

---

### Output
After all epochs complete:

```
outputs/
  khmer_ocr_10k_resnet/
    checkpoints/
      best_model.pth      ← the model with lowest CER (use this for inference)
      last_model.pth      ← the most recent epoch (use for resuming)
      epoch_005.pth       ← periodic snapshots
      metrics.csv         ← epoch-by-epoch training history
      experiment.json     ← summary: best CER, best epoch
```

**Slide bullet:**
> Output: `best_model.pth` — the model weights with the lowest Character Error Rate on validation data

---

## Summary Diagram for Slide

```
INPUT IMAGE (Khmer text image)
        │
        ▼
┌──────────────────────────────────────────┐
│         PREPROCESSING                    │
│  Normal: Resize → Normalize → Tensor    │
│  Skeleton: Resize → Thin strokes →      │
│            Tensor                        │
└──────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────┐
│      BATCH CHUNKING (DataLoader)         │
│  Group 64 images + encode labels         │
└──────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────┐
│      CNN BACKBONE                        │
│  CBC: 3 Conv blocks                      │
│  RBC: ResNet blocks (with skip connect.) │
│  Output: feature sequence (128 steps)    │
└──────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────┐
│      BiLSTM                              │
│  Read sequence forward + backward        │
│  Capture character context               │
└──────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────┐
│      LINEAR CLASSIFIER                   │
│  Predict character probability per step  │
└──────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────┐
│      CTC LOSS                            │
│  Compare prediction vs. true label       │
│  Allow flexible alignment                │
└──────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────┐
│      BACKPROPAGATION                     │
│  Update weights (Adam optimizer)         │
│  Repeat for N epochs                     │
└──────────────────────────────────────────┘
        │   (after every epoch)
        ▼
┌──────────────────────────────────────────┐
│      VALIDATION (CER)                    │
│  Measure error on unseen data            │
│  Save best_model.pth if improved         │
└──────────────────────────────────────────┘
        │
        ▼
OUTPUT: best_model.pth
```

---

## Key Differences Between the 4 Variants

| | CBC (Normal) | CBC-ZS (Skeleton) | RBC (Normal) | RBC-ZS (Skeleton) |
|---|---|---|---|---|
| **Preprocessing** | Resize + Normalize | Resize + Skeletonize | Resize + Normalize | Resize + Skeletonize |
| **CNN type** | Plain Conv blocks | Plain Conv blocks | Residual blocks | Residual blocks |
| **Skip connections** | No | No | Yes | Yes |
| **Input to model** | Original grayscale | Thinned 1px strokes | Original grayscale | Thinned 1px strokes |
| **What model learns** | Stroke shape + weight | Pure stroke structure | Stroke shape + weight | Pure stroke structure |
| **RNN / Decoder** | BiLSTM + CTC | BiLSTM + CTC | BiLSTM + CTC | BiLSTM + CTC |
| **Metric** | CER ↓ | CER ↓ | CER ↓ | CER ↓ |
