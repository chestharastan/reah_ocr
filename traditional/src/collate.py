import torch


def ocr_collate_fn(batch, vocab):
    images = []
    labels = []
    label_lengths = []

    for image, text in batch:
        images.append(image)

        encoded_text = vocab.encode(text)

        labels.extend(encoded_text)
        label_lengths.append(len(encoded_text))

    images = torch.stack(images)

    labels = torch.tensor(labels, dtype=torch.long)
    label_lengths = torch.tensor(label_lengths, dtype=torch.long)

    return images, labels, label_lengths


def ocr_collate_fn_attention(batch, vocab):
    """
    Collate for attention decoder. Produces padded target sequences:
      [SOS, char1, char2, ..., charN, EOS, PAD, PAD, ...]
    PAD token is 0 (same as <blank>, never appears in real labels).
    """
    images = []
    targets = []
    target_lengths = []

    for image, text in batch:
        images.append(image)
        encoded = vocab.encode(text)
        seq = [vocab.sos_id] + encoded + [vocab.eos_id]
        targets.append(seq)
        target_lengths.append(len(seq))

    images = torch.stack(images)

    max_len = max(target_lengths)
    padded = torch.zeros(len(targets), max_len, dtype=torch.long)
    for i, seq in enumerate(targets):
        padded[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)

    target_lengths = torch.tensor(target_lengths, dtype=torch.long)

    return images, padded, target_lengths