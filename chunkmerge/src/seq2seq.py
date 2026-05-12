import math
import torch
import torch.nn as nn

from chunking import chunk_batch


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class ChunkedSeq2Seq(nn.Module):
    """
    Generic chunked-encoder + transformer-decoder OCR model.

    The CNN backbone is supplied by the architecture file; this module owns the
    chunking, the patch projection, the transformer encoder/decoder, and
    autoregressive generation.

    Forward signature for training (teacher forcing):
        logits = model(images, tgt_in, tgt_pad_mask)

    Forward signature for inference:
        token_ids = model.generate(images, bos_id, eos_id, pad_id)

    Special token ids (pad/bos/eos) are passed at construction time so the
    decoder embedding and generation logic don't need to know about the vocab
    object directly.
    """

    def __init__(
        self,
        cnn,
        cnn_out_channels,
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
    ):
        super().__init__()
        self.cnn = cnn
        self.cnn_out_channels = cnn_out_channels
        self.d_model = d_model
        self.chunk_width = chunk_width
        self.overlap = overlap
        self.max_target_len = max_target_len
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id

        # LazyLinear so we don't need to know H' (height after CNN) up front.
        # Input size will be cnn_out_channels * H' on first forward pass.
        self.patch_proj = nn.LazyLinear(d_model)

        self.enc_pos = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.tgt_embed = nn.Embedding(num_classes, d_model, padding_idx=pad_id)
        self.dec_pos = PositionalEncoding(d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.classifier = nn.Linear(d_model, num_classes)

    def encode(self, images):
        """
        images: (B, C, H, W)
        returns memory: (B, N * W', d_model), where N is the number of chunks
        and W' is the per-chunk width after CNN downsampling. The merge across
        chunks happens here, before the decoder sees anything.
        """
        B = images.size(0)

        flat_chunks, N = chunk_batch(images, self.chunk_width, self.overlap)
        # flat_chunks: (B*N, C, H, chunk_w)
        feat = self.cnn(flat_chunks)  # (B*N, D, H', W')
        BN, D, Hp, Wp = feat.shape

        # (B*N, D, H', W') -> (B*N, W', D*H') -> (B*N, W', d_model)
        feat = feat.permute(0, 3, 1, 2).reshape(BN, Wp, D * Hp)
        feat = self.patch_proj(feat)
        feat = self.enc_pos(feat)
        feat = self.encoder(feat)  # (B*N, W', d_model)

        # merge chunks back into one long sequence per sample
        feat = feat.reshape(B, N * Wp, self.d_model)
        return feat

    def decode_step(self, memory, tgt_in, tgt_pad_mask=None):
        """
        memory: (B, S, d_model)
        tgt_in: (B, L) token ids, starting with BOS
        tgt_pad_mask: (B, L) bool, True at PAD positions
        returns logits: (B, L, num_classes)
        """
        L = tgt_in.size(1)
        tgt = self.tgt_embed(tgt_in) * math.sqrt(self.d_model)
        tgt = self.dec_pos(tgt)

        causal = nn.Transformer.generate_square_subsequent_mask(L).to(tgt.device)
        out = self.decoder(
            tgt,
            memory,
            tgt_mask=causal,
            tgt_key_padding_mask=tgt_pad_mask,
        )
        return self.classifier(out)

    def forward(self, images, tgt_in, tgt_pad_mask=None):
        memory = self.encode(images)
        return self.decode_step(memory, tgt_in, tgt_pad_mask)

    @torch.no_grad()
    def generate(self, images, max_len=None):
        """
        Greedy autoregressive decoding. Stops when every sample has emitted EOS
        or max_len is reached. Returns token ids including the leading BOS.
        """
        max_len = max_len or self.max_target_len
        device = images.device
        B = images.size(0)

        memory = self.encode(images)
        tgt = torch.full((B, 1), self.bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            logits = self.decode_step(memory, tgt)
            next_id = logits[:, -1].argmax(dim=-1)
            # once finished, keep emitting PAD so the sequence is well-formed
            next_id = torch.where(finished, torch.full_like(next_id, self.pad_id), next_id)
            tgt = torch.cat([tgt, next_id.unsqueeze(1)], dim=1)
            finished = finished | (next_id == self.eos_id)
            if bool(finished.all()):
                break

        return tgt
