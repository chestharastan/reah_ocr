import torch.nn as nn
from torchvision import transforms


class VggBiLstmCtc(nn.Module):
    """
    VGG-style backbone + BiLSTM + CTC classifier.

    Spatial flow for input (1, 64, 512):
        block1 : 2x [Conv(64) + BN + ReLU] + MaxPool(2,2)  → (64,  32, 256)
        block2 : 2x [Conv(128) + BN + ReLU] + MaxPool(2,2) → (128, 16, 128)
        block3 : 2x [Conv(256) + BN + ReLU]                → (256, 16, 128)

    RNN input: 128 time-steps of size 256*16=4096
    (same as ResNet/DenseNet so hidden_size/num_layers/dropout are identical)
    """

    def __init__(self, num_classes, hidden_size=256, num_layers=2, dropout=0.3):
        super().__init__()

        self.cnn = nn.Sequential(
            # block 1
            nn.Conv2d(1,  64, 3, padding=1, bias=False), nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # block 2
            nn.Conv2d(64,  128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # block 3
            nn.Conv2d(128, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
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
        features = self.cnn(x)                          # (B, 256, 16, W')

        batch_size, channels, height, width = features.size()
        features = features.permute(0, 3, 1, 2)         # (B, W', 256, 16)
        features = features.reshape(batch_size, width, channels * height)

        out, _ = self.rnn(features)
        return self.classifier(out)


Model = VggBiLstmCtc


def get_transform(image_height, image_width):
    return transforms.Compose([
        transforms.Resize((image_height, image_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
