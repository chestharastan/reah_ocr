import torch


def levenshtein_distance(a, b):
    """
    Calculate edit distance between two strings.
    Used for CER.
    """
    m = len(a)
    n = len(b)

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i

    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                cost = 0
            else:
                cost = 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    return dp[m][n]


def ctc_decode(pred_ids, blank_id=0):
    """
    Greedy CTC decoding.

    Example:
    [0, 5, 5, 0, 8, 8, 0] becomes [5, 8]
    """
    decoded = []
    previous_id = None

    for idx in pred_ids:
        idx = int(idx)

        if idx != blank_id and idx != previous_id:
            decoded.append(idx)

        previous_id = idx

    return decoded


def ids_to_text(ids, vocab):
    """
    Convert token IDs back to Khmer text.
    """
    chars = []

    for idx in ids:
        if hasattr(vocab, "idx_to_char"):
            if isinstance(vocab.idx_to_char, dict):
                chars.append(vocab.idx_to_char[idx])
            else:
                chars.append(vocab.idx_to_char[idx])
        else:
            raise AttributeError("vocab must have idx_to_char")

    return "".join(chars)


def validate_one_epoch(model, dataloader, vocab, device, blank_id=0):
    model.eval()

    total_distance = 0
    total_characters = 0

    with torch.no_grad():
        for images, labels, label_lengths in dataloader:
            images = images.to(device)

            outputs = model(images)

            # outputs shape: [batch, time, num_classes]
            pred_ids = outputs.argmax(dim=2)

            label_start = 0

            for i in range(images.size(0)):
                label_length = label_lengths[i].item()

                true_ids = labels[label_start:label_start + label_length].tolist()
                label_start += label_length

                pred_sequence = pred_ids[i].cpu().tolist()
                pred_decoded_ids = ctc_decode(
                    pred_sequence,
                    blank_id=blank_id
                )

                true_text = ids_to_text(true_ids, vocab)
                pred_text = ids_to_text(pred_decoded_ids, vocab)

                distance = levenshtein_distance(pred_text, true_text)

                total_distance += distance
                total_characters += len(true_text)

    if total_characters == 0:
        return 1.0

    cer = total_distance / total_characters

    return cer