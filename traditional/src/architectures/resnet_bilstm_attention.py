import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


# -----------------------------------------------------------------------------
# ResNet backbone (identical to resnet_bilstm_ctc)
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Bahdanau attention decoder
# -----------------------------------------------------------------------------
class _BahdanauAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.W_encoder = nn.Linear(encoder_dim, attention_dim)
        self.W_decoder = nn.Linear(decoder_dim, attention_dim)
        self.v = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, encoder_out, decoder_hidden):
        energy = self.v(torch.tanh(
            self.W_encoder(encoder_out) +
            self.W_decoder(decoder_hidden).unsqueeze(1)
        )).squeeze(2)
        weights = F.softmax(energy, dim=1)
        context = (weights.unsqueeze(2) * encoder_out).sum(dim=1)
        return context, weights


class _AttentionDecoder(nn.Module):
    def __init__(self, num_classes, encoder_dim, decoder_dim, attention_dim, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, decoder_dim, padding_idx=0)
        self.attention = _BahdanauAttention(encoder_dim, decoder_dim, attention_dim)
        self.rnn = nn.GRUCell(decoder_dim + encoder_dim, decoder_dim)
        self.classifier = nn.Linear(decoder_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def step(self, encoder_out, prev_token, hidden):
        embedded = self.dropout(self.embedding(prev_token))
        context, _ = self.attention(encoder_out, hidden)
        hidden = self.rnn(torch.cat([embedded, context], dim=1), hidden)
        return self.classifier(hidden), hidden


# -----------------------------------------------------------------------------
# ResNet encoder + BiLSTM + Bahdanau attention decoder
# -----------------------------------------------------------------------------
class ResNetBiLstmAttention(nn.Module):
    """
    ResNet backbone + BiLSTM encoder + Bahdanau attention GRU decoder.

    Spatial flow for input (1, 64, 512):
        stem   : (1,  64, 512) -> (64,  32, 256)  MaxPool 2x2
        stage1 : (64, 32, 256) -> (64,  32, 256)  stride=1
        stage2 : (64, 32, 256) -> (128, 16, 128)  stride=2
        stage3 : (128,16, 128) -> (256, 16, 128)  stride=1
        reshape:                -> (B, 128, 4096)
        BiLSTM :                -> (B, 128, hidden*2)   encoder_out
    """

    def __init__(self, num_classes, hidden_size=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.num_classes = num_classes
        self.sos_id = num_classes - 2
        self.eos_id = num_classes - 1

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

        encoder_dim = hidden_size * 2
        decoder_dim = hidden_size
        self.encoder_to_decoder = nn.Linear(encoder_dim, decoder_dim)
        self.decoder = _AttentionDecoder(num_classes, encoder_dim, decoder_dim, hidden_size, dropout)

    def _encode(self, images):
        x = self.cnn(images)                            # (B, 256, 16, W')
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2).reshape(b, w, c * h)
        encoder_out, _ = self.rnn(x)
        return encoder_out                              # (B, T, encoder_dim)

    def forward(self, images, targets=None, max_len=50):
        encoder_out = self._encode(images)
        init_hidden = self.encoder_to_decoder(encoder_out.mean(dim=1))

        if targets is not None:
            logits, hidden = [], init_hidden
            for t in range(targets.size(1) - 1):
                logit, hidden = self.decoder.step(encoder_out, targets[:, t], hidden)
                logits.append(logit)
            return torch.stack(logits, dim=1)
        else:
            batch_size = encoder_out.size(0)
            prev = torch.full((batch_size,), self.sos_id, dtype=torch.long, device=images.device)
            hidden, logits = init_hidden, []
            for _ in range(max_len):
                logit, hidden = self.decoder.step(encoder_out, prev, hidden)
                logits.append(logit)
                prev = logit.argmax(dim=1)
                if (prev == self.eos_id).all():
                    break
            return torch.stack(logits, dim=1)


Model = ResNetBiLstmAttention


def get_transform(image_height, image_width):
    return transforms.Compose([
        transforms.Resize((image_height, image_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
