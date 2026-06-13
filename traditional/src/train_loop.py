import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    scaler = GradScaler(enabled=device.type == "cuda")
    total_loss = 0.0

    for images, labels, label_lengths in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        label_lengths = label_lengths.to(device)

        with autocast(enabled=device.type == "cuda"):
            outputs = model(images)
            outputs = outputs.permute(1, 0, 2)
            log_probs = torch.nn.functional.log_softmax(outputs, dim=2)

            batch_size = images.size(0)
            input_lengths = torch.full(
                size=(batch_size,),
                fill_value=log_probs.size(0),
                dtype=torch.long,
                device=device,
            )

            loss = criterion(log_probs, labels, input_lengths, label_lengths)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(dataloader)