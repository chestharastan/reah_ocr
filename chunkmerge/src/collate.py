import torch


def ocr_collate_fn(batch, vocab):
    """
    Build seq2seq targets:
        tgt_in  = [BOS, c0, c1, ..., cN-1] padded with PAD
        tgt_out = [c0,  c1, ..., cN-1, EOS] padded with PAD

    The decoder consumes tgt_in (teacher forcing) and the loss is computed
    against tgt_out, with PAD positions ignored.
    """
    images = []
    encoded_texts = []

    for image, text in batch:
        images.append(image)
        encoded_texts.append(vocab.encode(text))

    images = torch.stack(images)

    # +1 for BOS or EOS
    max_len = max(len(t) for t in encoded_texts) + 1
    B = len(encoded_texts)

    tgt_in = torch.full((B, max_len), vocab.pad_id, dtype=torch.long)
    tgt_out = torch.full((B, max_len), vocab.pad_id, dtype=torch.long)

    for i, ids in enumerate(encoded_texts):
        tgt_in[i, 0] = vocab.bos_id
        tgt_in[i, 1 : 1 + len(ids)] = torch.tensor(ids, dtype=torch.long)

        tgt_out[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        tgt_out[i, len(ids)] = vocab.eos_id

    tgt_pad_mask = tgt_in == vocab.pad_id

    return images, tgt_in, tgt_out, tgt_pad_mask
