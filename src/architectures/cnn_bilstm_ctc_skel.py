import numpy as np
import torch.nn as nn
from PIL import Image
from skimage.morphology import skeletonize
from torchvision import transforms


class SkeletonTransform:
    """
    Skeletonize black text on white background using Zhang-Suen thinning (scikit-image).
    Converts strokes to 1-pixel-wide centerlines.
    """
    def __call__(self, img):
        arr = np.array(img.convert("L"))
        # text pixels are dark (< 128) → True for foreground
        binary = arr < 128
        skel = skeletonize(binary).astype(np.uint8) * 255
        # invert back: black text on white background
        skel = 255 - skel
        return Image.fromarray(skel, mode="L")


class Model(nn.Module):
    """
    CNN-BiLSTM-CTC trained on skeletonized Khmer text images.

    Skeletonization strips stroke thickness and leaves only the 1-pixel
    centerline of each character stroke, so the model learns pure shape
    rather than thickness variations.

    Architecture is the same as cnn_bilstm_ctc — the improvement comes
    entirely from the cleaner input representation.
    """

    def __init__(self, num_classes, hidden_size=256, num_layers=2, dropout=0.3):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.rnn = nn.LSTM(
            input_size=256 * 16,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        features = self.cnn(x)

        batch_size, channels, height, width = features.size()

        features = features.permute(0, 3, 1, 2)
        features = features.reshape(batch_size, width, channels * height)

        sequence_output, _ = self.rnn(features)
        output = self.classifier(sequence_output)

        return output


def get_transform(image_height, image_width):
    return transforms.Compose([
        transforms.Resize((image_height, image_width)),
        SkeletonTransform(),
        transforms.ToTensor(),
    ])
