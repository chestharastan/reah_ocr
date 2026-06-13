import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


# -----------------------------------------------------------------------------
# VGG backbone (identical to vgg_bilstm_ctc)
# -----------------------------------------------------------------------------
class VggBiLstmAttention(nn.Module):
    """
    VGG-style backbone + BiLSTM encoder + Bahdanau attention GRU decoder.

    Spatial flow for input (1, 64, 512):
        block1 : 2x [Conv(64) + BN + ReLU] + MaxPool(2,2)  → (64,  32, 256)
        block2 : 2x [Conv(128) + BN + ReLU] + MaxPool(2,2) → (128, 16, 128)
        block3 : 2x [Conv(256) + BN + ReLU]                → (256, 16, 128)
        reshape :                                           → (B, 128, 4096)
        BiLSTM  :                                           → (B, 128, hidden*2)
    """

    def __init__(self, num_classes, hidden_size=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.num_classes = num_classes
        self.sos_id = num_classes - 2
        self.eos_id = num_classes - 1

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


# -----------------------------------------------------------------------------
# Bahdanau attention decoder (shared across all attention architectures)
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


Model = VggBiLstmAttention


def get_transform(image_height, image_width):
    return transforms.Compose([
        transforms.Resize((image_height, image_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
