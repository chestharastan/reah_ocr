import torch.nn as nn
from torchvision import transforms

from seq2seq import ChunkedSeq2Seq


class _PaperChunkCNN(nn.Module):
    """
    Small CNN matching the paper: x4 downsampling in both spatial dims.

    For input (1, 48, 100): output is (cnn_channels, 12, 25).
    """

    def __init__(self, in_channels=1, out_channels=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


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
):
    cnn = _PaperChunkCNN(in_channels=1, out_channels=cnn_channels)
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
