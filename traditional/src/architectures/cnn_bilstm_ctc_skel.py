import numpy as np
import torch.nn as nn
from PIL import Image
import cv2
from torchvision import transforms


class SkeletonTransform:
    """
    Skeletonize black text on white background using morphological thinning (OpenCV).
    Converts strokes to 1-pixel-wide centerlines.
    """
    def __call__(self, img):
        arr = np.array(img.convert("L"))
        # invert: dark text → white foreground for morphology
        _, binary = cv2.threshold(arr, 128, 255, cv2.THRESH_BINARY_INV)

        skel = np.zeros_like(binary)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        temp = binary.copy()

        while True:
            eroded = cv2.erode(temp, element)
            opened = cv2.dilate(eroded, element)
            subset = cv2.subtract(temp, opened)
            skel = cv2.bitwise_or(skel, subset)
            temp = eroded.copy()
            if cv2.countNonZero(temp) == 0:
                break

        # invert back: black text on white background
        skel = cv2.bitwise_not(skel)
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
