import numpy as np
import torch.nn as nn
import cv2
from PIL import Image
from torchvision import transforms

from seq2seq import ChunkedSeq2Seq


class SkeletonTransform:
    """
    Skeletonize black text on white background using morphological thinning.
    Strips stroke thickness so the model learns pure shape rather than
    thickness variations. Mirrors traditional/cnn_bilstm_ctc_skel.
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


class _PaperChunkCNN(nn.Module):
    """
    Same backbone as chunked_paper_seq2seq — the only change is the input
    pre-processing (skeletonization) applied via get_transform.
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
        SkeletonTransform(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
