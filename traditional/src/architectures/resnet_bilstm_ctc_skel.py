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

        skel = cv2.bitwise_not(skel)
        return Image.fromarray(skel, mode="L")


class _BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        skip = x if self.downsample is None else self.downsample(x)
        return self.relu(self.conv(x) + skip)


def _make_stage(in_channels, out_channels, num_blocks, stride=1):
    layers = [_BasicBlock(in_channels, out_channels, stride=stride)]
    for _ in range(1, num_blocks):
        layers.append(_BasicBlock(out_channels, out_channels))
    return nn.Sequential(*layers)


class Model(nn.Module):
    """
    ResNet backbone + BiLSTM + CTC trained on skeletonized Khmer text images.

    Skeletonization strips stroke thickness to 1-pixel centerlines so the
    model learns pure shape. The ResNet backbone is identical to
    resnet_bilstm_ctc — the improvement comes from the cleaner input.

    Spatial flow (image_height=64, image_width=512):
        stem   : (1,  64, 512) -> (64,  32, 256)
        stage1 : (64, 32, 256) -> (64,  32, 256)
        stage2 : (64, 32, 256) -> (128, 16, 128)
        stage3 : (128,16, 128) -> (256, 16, 128)
        reshape:               -> (B, 128, 4096)
    """

    def __init__(self, num_classes, hidden_size=256, num_layers=2, dropout=0.3):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            _make_stage(64,  64,  num_blocks=1, stride=1),
            _make_stage(64,  128, num_blocks=1, stride=2),
            _make_stage(128, 256, num_blocks=1, stride=1),
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
        return self.classifier(sequence_output)


def get_transform(image_height, image_width):
    return transforms.Compose([
        transforms.Resize((image_height, image_width)),
        SkeletonTransform(),
        transforms.ToTensor(),
    ])
