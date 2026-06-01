import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


# -----------------------------------------------------------------------------
# DenseNet backbone (identical to densenet_bilstm_ctc)
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Bahdanau attention decoder (identical to cnn_bilstm_attention)
# -----------------------------------------------------------------------------
class BahdanauAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.W_encoder = nn.Linear(encoder_dim, attention_dim)
        self.W_decoder = nn.Linear(decoder_dim, attention_dim)
        self.v = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, encoder_out, decoder_hidden):
        # encoder_out: [batch, T, encoder_dim]
        # decoder_hidden: [batch, decoder_dim]
        energy = self.v(torch.tanh(
            self.W_encoder(encoder_out) +
            self.W_decoder(decoder_hidden).unsqueeze(1)
        )).squeeze(2)  # [batch, T]
        weights = F.softmax(energy, dim=1)
        context = (weights.unsqueeze(2) * encoder_out).sum(dim=1)  # [batch, encoder_dim]
        return context, weights


class AttentionDecoder(nn.Module):
    def __init__(self, num_classes, encoder_dim, decoder_dim, attention_dim, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, decoder_dim, padding_idx=0)
        self.attention = BahdanauAttention(encoder_dim, decoder_dim, attention_dim)
        self.rnn = nn.GRUCell(decoder_dim + encoder_dim, decoder_dim)
        self.classifier = nn.Linear(decoder_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def step(self, encoder_out, prev_token, hidden):
        embedded = self.dropout(self.embedding(prev_token))  # [batch, decoder_dim]
        context, _ = self.attention(encoder_out, hidden)     # [batch, encoder_dim]
        rnn_input = torch.cat([embedded, context], dim=1)
        hidden = self.rnn(rnn_input, hidden)
        logit = self.classifier(hidden)                      # [batch, num_classes]
        return logit, hidden


# -----------------------------------------------------------------------------
# DenseNet encoder + BiLSTM + Bahdanau attention decoder
# -----------------------------------------------------------------------------
class DenseNetBiLstmAttention(nn.Module):
    """
    DenseNet CNN backbone → BiLSTM encoder → Bahdanau attention GRU decoder.

    DenseNet encoder is identical to densenet_bilstm_ctc; the attention decoder
    is identical to cnn_bilstm_attention. Spatial flow (h=64, w=512):
      stem        : (B,   1, 64, 512) → (B,  64, 32, 256)   MaxPool 2×2
      dense1      : (B,  64, 32, 256) → (B, 192, 32, 256)   4 layers, k=32
      transition1 : (B, 192, 32, 256) → (B,  96, 16, 128)   AvgPool 2×2
      dense2      : (B,  96, 16, 128) → (B, 288, 16, 128)   6 layers, k=32
      proj        : (B, 288, 16, 128) → (B, 256, 16, 128)   1×1 conv
      reshape     :                  → (B, 128, 4096)        width = time axis
      BiLSTM      :                  → (B, 128, hidden*2)    encoder_out
    """

    def __init__(self, num_classes, hidden_size=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.num_classes = num_classes
        # SOS and EOS occupy the last two slots in KhmerVocabAttention
        self.sos_id = num_classes - 2
        self.eos_id = num_classes - 1

        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.dense1 = _DenseBlock(num_layers=4, in_channels=64, growth_rate=32)
        self.trans1 = _Transition(in_channels=192, out_channels=96)
        self.dense2 = _DenseBlock(num_layers=6, in_channels=96, growth_rate=32)

        self.proj = nn.Sequential(
            nn.BatchNorm2d(288),
            nn.ReLU(inplace=True),
            nn.Conv2d(288, 256, kernel_size=1, bias=False),
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
        self.decoder = AttentionDecoder(
            num_classes, encoder_dim, decoder_dim, hidden_size, dropout
        )

    def _encode(self, images):
        x = self.stem(images)
        x = self.dense1(x)
        x = self.trans1(x)
        x = self.dense2(x)
        x = self.proj(x)

        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2).reshape(b, w, c * h)
        encoder_out, _ = self.rnn(x)
        return encoder_out  # [batch, T, encoder_dim]

    def forward(self, images, targets=None, max_len=50):
        encoder_out = self._encode(images)
        init_hidden = self.encoder_to_decoder(encoder_out.mean(dim=1))

        if targets is not None:
            # Teacher forcing: targets is [batch, seq_len] starting with SOS
            logits = []
            hidden = init_hidden
            for t in range(targets.size(1) - 1):
                logit, hidden = self.decoder.step(encoder_out, targets[:, t], hidden)
                logits.append(logit)
            return torch.stack(logits, dim=1)  # [batch, seq_len-1, num_classes]
        else:
            # Greedy decode until EOS or max_len
            batch_size = encoder_out.size(0)
            prev = torch.full(
                (batch_size,), self.sos_id, dtype=torch.long, device=images.device
            )
            hidden = init_hidden
            logits = []
            for _ in range(max_len):
                logit, hidden = self.decoder.step(encoder_out, prev, hidden)
                logits.append(logit)
                prev = logit.argmax(dim=1)
                if (prev == self.eos_id).all():
                    break
            return torch.stack(logits, dim=1)  # [batch, decoded_len, num_classes]


Model = DenseNetBiLstmAttention


def get_transform(image_height, image_width):
    return transforms.Compose([
        transforms.Resize((image_height, image_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
