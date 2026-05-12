import torch
import torch.nn as nn
from torchvision import transforms


class _DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False),
            nn.BatchNorm2d(4 * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x):
        return torch.cat([x, self.layer(x)], dim=1)


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(_DenseLayer(in_channels + i * growth_rate, growth_rate))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class _Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.block(x)


class DenseNetBiLstmBiGruCtc(nn.Module):
    """
    DenseNet CNN backbone → BiLSTM → BiGRU → CTC classifier.

    Spatial flow (image_height=64, image_width=512):
      stem        : (B,  1, 64, 512) → (B,  64, 32, 256)   MaxPool 2×2
      dense1      : (B,  64, 32, 256) → (B, 192, 32, 256)  4 layers, k=32
      transition1 : (B, 192, 32, 256) → (B,  96, 16, 128)  compress + AvgPool 2×2
      dense2      : (B,  96, 16, 128) → (B, 288, 16, 128)  6 layers, k=32
      proj        : (B, 288, 16, 128) → (B, 256, 16, 128)  1×1 conv
      reshape     : → (B, 128, 4096)   width becomes time axis
      BiLSTM      : → (B, 128, hidden*2)
      BiGRU       : → (B, 128, hidden*2)
      classifier  : → (B, 128, num_classes)
    """

    def __init__(self, num_classes, hidden_size=256, num_layers=2, dropout=0.3):
        super().__init__()

        # stem: initial feature extraction + first spatial downsampling
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # dense block 1: 64 → 192 channels  (4 layers × growth_rate 32)
        self.dense1 = _DenseBlock(num_layers=4, in_channels=64, growth_rate=32)
        # transition 1: 192 → 96 channels + second spatial downsampling
        self.trans1 = _Transition(in_channels=192, out_channels=96)

        # dense block 2: 96 → 288 channels  (6 layers × growth_rate 32)
        self.dense2 = _DenseBlock(num_layers=6, in_channels=96, growth_rate=32)

        # project to a fixed channel count before feeding into RNN
        self.proj = nn.Sequential(
            nn.BatchNorm2d(288),
            nn.ReLU(inplace=True),
            nn.Conv2d(288, 256, kernel_size=1, bias=False),
        )

        # BiLSTM: captures long-range horizontal dependencies
        self.bilstm = nn.LSTM(
            input_size=256 * 16,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.dropout = nn.Dropout(p=dropout)

        # BiGRU: efficient refinement layer on top of BiLSTM features
        self.bigru = nn.GRU(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.dense1(x)
        x = self.trans1(x)
        x = self.dense2(x)
        x = self.proj(x)

        batch_size, channels, height, width = x.size()
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch_size, width, channels * height)

        x, _ = self.bilstm(x)
        x = self.dropout(x)
        x, _ = self.bigru(x)
        x = self.classifier(x)

        return x


Model = DenseNetBiLstmBiGruCtc


def get_transform(image_height, image_width):
    return transforms.Compose([
        transforms.Resize((image_height, image_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
