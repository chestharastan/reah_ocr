import torch.nn as nn
from torchvision import transforms

from seq2seq import ChunkedSeq2Seq


class _BasicBlock(nn.Module):
    """ResNet basic block: two 3×3 convs with a skip connection."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        # 1×1 projection when shape changes (stride or channels)
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


class _ResNetChunkCNN(nn.Module):
    """
    Lightweight ResNet backbone for per-chunk feature extraction.

    Spatial flow for input (1, 48, 100):
        stem   : (1, 48, 100)  -> (64, 24, 50)
        stage1 : (64, 24, 50)  -> (64, 24, 50)   stride=1
        stage2 : (64, 24, 50)  -> (128, 12, 25)  stride=2
        stage3 : (128, 12, 25) -> (256, 12, 25)  stride=1
        proj   : (256, 12, 25) -> (out_channels, 12, 25)
    """

    def __init__(self, in_channels=1, out_channels=256, layers=(1, 1, 1)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.stage1 = _make_stage(64,  64,  layers[0], stride=1)
        self.stage2 = _make_stage(64,  128, layers[1], stride=2)
        self.stage3 = _make_stage(128, 256, layers[2], stride=1)
        self.proj = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channels, kernel_size=1, bias=False),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.proj(x)


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
    resnet_layers=(1, 1, 1),
):
    cnn = _ResNetChunkCNN(
        in_channels=1,
        out_channels=cnn_channels,
        layers=tuple(resnet_layers),
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
