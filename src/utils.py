import csv
import json
import os
import yaml
import torch
from datetime import datetime


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


def init_experiment_log(checkpoint_dir, config):
    os.makedirs(checkpoint_dir, exist_ok=True)

    csv_path = os.path.join(checkpoint_dir, "metrics.csv")
    # Only write the header if the file doesn't exist yet (fresh run).
    # On resume, keep existing rows and just append new epochs.
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_cer", "lr", "epoch_time_s"])

    json_path = os.path.join(checkpoint_dir, "experiment.json")
    # On resume, keep the original started_at from the existing summary.
    existing_started_at = None
    if os.path.exists(json_path):
        with open(json_path) as f:
            existing_started_at = json.load(f).get("started_at")

    summary = {
        "architecture": config["model"]["architecture"],
        "started_at": existing_started_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": config,
        "best_cer": None,
        "best_epoch": None,
        "finished_at": None,
    }
    json_path = os.path.join(checkpoint_dir, "experiment.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return csv_path, json_path


def log_epoch(csv_path, epoch, train_loss, val_cer, lr, epoch_time=None):
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, f"{train_loss:.6f}", f"{val_cer:.6f}", f"{lr:.2e}",
                         f"{epoch_time:.1f}" if epoch_time is not None else ""])


def finish_experiment_log(json_path, best_cer, best_epoch):
    with open(json_path, "r") as f:
        summary = json.load(f)

    finished_at = datetime.now()
    started_at = datetime.strptime(summary["started_at"], "%Y-%m-%d %H:%M:%S")
    elapsed = int((finished_at - started_at).total_seconds())
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)

    summary["best_cer"] = round(best_cer, 6)
    summary["best_epoch"] = best_epoch
    summary["finished_at"] = finished_at.strftime("%Y-%m-%d %H:%M:%S")
    summary["time_trained"] = f"{h}h {m:02d}m {s:02d}s"

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)