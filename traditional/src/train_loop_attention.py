import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm


def train_one_epoch_attention(model, dataloader, optimizer, device):
    model.train()
    scaler = GradScaler(enabled=device.type == "cuda")
    total_loss = 0.0

    for images, targets, target_lengths in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        targets = targets.to(device)

        with autocast(enabled=device.type == "cuda"):
            logits = model(images, targets=targets)  # [batch, max_len-1, num_classes]

            batch_size, seq_len, num_classes = logits.shape
            logits_flat = logits.reshape(-1, num_classes)
            labels_flat = targets[:, 1:seq_len + 1].reshape(-1)

            mask = labels_flat != 0
            loss = F.cross_entropy(logits_flat[mask], labels_flat[mask])

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(dataloader)
