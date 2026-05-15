import torch.nn as nn
from torchvision import transforms


class _BasicBlock(nn.Module):
    """ResNet basic block: two 3×3 convs with a residual skip connection."""

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


class ResNetBiLstmCtc(nn.Module):
    """
    ResNet backbone + BiLSTM + CTC, drop-in replacement for CnnBiLstmCtc.

    Spatial flow for input (1, 64, 512):
        stem   : (1,  64, 512) -> (64,  32, 256)  MaxPool 2×2
        stage1 : (64, 32, 256) -> (64,  32, 256)  stride=1
        stage2 : (64, 32, 256) -> (128, 16, 128)  stride=2
        stage3 : (128,16, 128) -> (256, 16, 128)  stride=1

    RNN input: width=128 time-steps, each of size 256*16=4096
    (same as CnnBiLstmCtc so hidden_size / num_layers / dropout are identical)
    """

    def __init__(self, num_classes, hidden_size=256, num_layers=2, dropout=0.3):
        super().__init__()

        self.cnn = nn.Sequential(
            # stem
            nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # stage 1 — keep spatial size
            _make_stage(64,  64,  num_blocks=1, stride=1),
            # stage 2 — halve height & width
            _make_stage(64,  128, num_blocks=1, stride=2),
            # stage 3 — deepen channels only
            _make_stage(128, 256, num_blocks=1, stride=1),
        )

        self.rnn = nn.LSTM(
            input_size=256 * 16,   # 256 channels × 16 height (64 // 4)
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        features = self.cnn(x)                          # (B, 256, 16, W')

        batch_size, channels, height, width = features.size()
        features = features.permute(0, 3, 1, 2)         # (B, W', 256, 16)
        features = features.reshape(batch_size, width, channels * height)

        sequence_output, _ = self.rnn(features)
        return self.classifier(sequence_output)


Model = ResNetBiLstmCtc


def get_transform(image_height, image_width):
    return transforms.Compose([
        transforms.Resize((image_height, image_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
