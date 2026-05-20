import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


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


class CnnBiLstmAttention(nn.Module):
    def __init__(self, num_classes, hidden_size=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.num_classes = num_classes
        # SOS and EOS occupy the last two slots in KhmerVocabAttention
        self.sos_id = num_classes - 2
        self.eos_id = num_classes - 1

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
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
        features = self.cnn(images)
        b, c, h, w = features.size()
        features = features.permute(0, 3, 1, 2).reshape(b, w, c * h)
        encoder_out, _ = self.rnn(features)
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


Model = CnnBiLstmAttention


def get_transform(image_height, image_width):
    return transforms.Compose([
        transforms.Resize((image_height, image_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
