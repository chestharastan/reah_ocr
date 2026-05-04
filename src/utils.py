import os
import yaml
import torch


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def load_checkpoint(path, model, optimizer, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    print(f"Resumed from epoch {checkpoint['epoch']}, val_cer={checkpoint['val_cer']:.4f}")
    return checkpoint["epoch"] + 1


def save_checkpoint(model, optimizer, epoch, val_cer, checkpoint_dir, save_epoch_file=True):
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_cer": val_cer,
    }

    last_path = os.path.join(checkpoint_dir, "last_model.pth")
    torch.save(checkpoint, last_path)

    if save_epoch_file:
        epoch_path = os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.pth")
        torch.save(checkpoint, epoch_path)


def save_best_model(model, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_path = os.path.join(checkpoint_dir, "best_model.pth")
    torch.save(model.state_dict(), best_path)