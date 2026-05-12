import torch


def levenshtein_distance(a, b):
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[m][n]


def _ground_truth_text(tgt_out_row, vocab):
    """Strip pad/eos from a target-output row to recover the original text."""
    chars = []
    for idx in tgt_out_row.tolist():
        if idx == vocab.eos_id:
            break
        if idx == vocab.pad_id:
            continue
        chars.append(vocab.idx_to_char[idx])
    return "".join(chars)


def validate_one_epoch(model, dataloader, vocab, device, max_len=None):
    model.eval()
    total_distance = 0
    total_characters = 0

    with torch.no_grad():
        for images, _tgt_in, tgt_out, _tgt_pad_mask in dataloader:
            images = images.to(device)

            pred_ids = model.generate(images, max_len=max_len)  # (B, L) includes BOS

            for i in range(images.size(0)):
                # drop the leading BOS before decoding
                pred_text = vocab.decode(pred_ids[i, 1:].cpu().tolist())
                true_text = _ground_truth_text(tgt_out[i], vocab)

                total_distance += levenshtein_distance(pred_text, true_text)
                total_characters += len(true_text)

    if total_characters == 0:
        return 1.0
    return total_distance / total_characters
