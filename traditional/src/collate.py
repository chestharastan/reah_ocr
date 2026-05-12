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