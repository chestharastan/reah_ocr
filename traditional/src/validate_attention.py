import torch
from validate import levenshtein_distance, ids_to_text


def validate_one_epoch_attention(model, dataloader, vocab, device, max_len=50):
    model.eval()
    total_char_dist = 0
    total_chars = 0
    total_word_dist = 0
    total_words = 0

    with torch.no_grad():
        for images, targets, target_lengths in dataloader:
            images = images.to(device)

            logits = model(images, max_len=max_len)
            pred_ids = logits.argmax(dim=2)

            for i in range(images.size(0)):
                pred_seq = []
                for idx in pred_ids[i].cpu().tolist():
                    if idx == vocab.eos_id:
                        break
                    if idx not in (vocab.sos_id, vocab.eos_id, 0):
                        pred_seq.append(idx)

                true_seq = []
                for j in range(target_lengths[i].item()):
                    idx = targets[i][j].item()
                    if idx not in (vocab.sos_id, vocab.eos_id, 0):
                        true_seq.append(idx)

                true_text = ids_to_text(true_seq, vocab)
                pred_text = ids_to_text(pred_seq, vocab)

                total_char_dist += levenshtein_distance(pred_text, true_text)
                total_chars += len(true_text)

                true_words = true_text.split()
                pred_words = pred_text.split()
                total_word_dist += levenshtein_distance(pred_words, true_words)
                total_words += len(true_words)

    cer = total_char_dist / total_chars if total_chars > 0 else 1.0
    wer = total_word_dist / total_words if total_words > 0 else 1.0

    return cer, wer
