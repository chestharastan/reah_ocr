import torch
from tqdm import tqdm


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for images, tgt_in, tgt_out, tgt_pad_mask in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        tgt_in = tgt_in.to(device)
        tgt_out = tgt_out.to(device)
        tgt_pad_mask = tgt_pad_mask.to(device)

        logits = model(images, tgt_in, tgt_pad_mask)  # (B, L, V)

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            tgt_out.reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
