import torch
from validate import levenshtein_distance, ids_to_text


def validate_one_epoch_attention(model, dataloader, vocab, device, max_len=50):
    model.eval()
    total_distance = 0
    total_characters = 0

    with torch.no_grad():
        for images, targets, target_lengths in dataloader:
            images = images.to(device)

            # Greedy decode: returns [batch, decoded_len, num_classes]
            logits = model(images, max_len=max_len)
            pred_ids = logits.argmax(dim=2)  # [batch, decoded_len]

            for i in range(images.size(0)):
                # Collect predicted characters, stop at EOS
                pred_seq = []
                for idx in pred_ids[i].cpu().tolist():
                    if idx == vocab.eos_id:
                        break
                    if idx not in (vocab.sos_id, vocab.eos_id, 0):
                        pred_seq.append(idx)

                # Collect true characters from padded target (skip SOS/EOS/PAD)
                true_seq = []
                for j in range(target_lengths[i].item()):
                    idx = targets[i][j].item()
                    if idx not in (vocab.sos_id, vocab.eos_id, 0):
                        true_seq.append(idx)

                true_text = ids_to_text(true_seq, vocab)
                pred_text = ids_to_text(pred_seq, vocab)

                total_distance += levenshtein_distance(pred_text, true_text)
                total_characters += len(true_text)

    if total_characters == 0:
        return 1.0

    return total_distance / total_characters
