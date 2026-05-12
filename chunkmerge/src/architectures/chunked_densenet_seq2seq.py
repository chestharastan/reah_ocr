import torch
import torch.nn as nn
from torchvision import transforms

from seq2seq import ChunkedSeq2Seq


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
        self.layers = nn.ModuleList(
            [_DenseLayer(in_channels + i * growth_rate, growth_rate) for i in range(num_layers)]
        )

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


class _DenseNetChunkCNN(nn.Module):
    """
    DenseNet-style backbone, matched to traditional/densenet but used as a
    per-chunk feature extractor.

    Spatial flow for input (1, 48, 100):
        stem        : (1, 48, 100) -> (64, 24, 50)
        dense1      : (64, 24, 50) -> (192, 24, 50)
        transition1 : (192, 24, 50) -> (96, 12, 25)
        dense2      : (96, 12, 25) -> (288, 12, 25)
        proj        : (288, 12, 25) -> (out_channels, 12, 25)
    """

    def __init__(self, in_channels=1, out_channels=256, growth_rate=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.dense1 = _DenseBlock(num_layers=4, in_channels=64, growth_rate=growth_rate)
        self.trans1 = _Transition(in_channels=64 + 4 * growth_rate, out_channels=96)
        self.dense2 = _DenseBlock(num_layers=6, in_channels=96, growth_rate=growth_rate)
        self.proj = nn.Sequential(
            nn.BatchNorm2d(96 + 6 * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(96 + 6 * growth_rate, out_channels, kernel_size=1, bias=False),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.dense1(x)
        x = self.trans1(x)
        x = self.dense2(x)
        x = self.proj(x)
        return x


def Model(
    num_classes,
    pad_id,
    bos_id,
    eos_id,
    d_model=256,
    nhead=8,
    num_encoder_layers=4,
    num_decoder_layers=4,
    dim_feedforward=1024,
    dropout=0.1,
    chunk_width=100,
    overlap=16,
    max_target_len=256,
    cnn_channels=256,
    growth_rate=32,
):
    cnn = _DenseNetChunkCNN(
        in_channels=1,
        out_channels=cnn_channels,
        growth_rate=growth_rate,
    )
    return ChunkedSeq2Seq(
        cnn=cnn,
        cnn_out_channels=cnn_channels,
        num_classes=num_classes,
        pad_id=pad_id,
        bos_id=bos_id,
        eos_id=eos_id,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        chunk_width=chunk_width,
        overlap=overlap,
        max_target_len=max_target_len,
    )


def get_transform(image_height, image_width):
    return transforms.Compose([
        transforms.Resize((image_height, image_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
