import os
import sys
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from functools import partial

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils import (load_config, load_checkpoint, save_checkpoint, save_best_model,
                   init_experiment_log, log_epoch, finish_experiment_log)
from dataset import OCRDataset
from vocab import KhmerVocab
from collate import ocr_collate_fn
from architectures import build_model, build_transform
from train_loop import train_one_epoch
from validate import validate_one_epoch


DEFAULT_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "..",
    "setting",
    "config.yml"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Khmer OCR model")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to checkpoint to resume from"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    # -----------------------------
    # Device
    # -----------------------------
    if torch.cuda.is_available() and config["training"]["device"] == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    # -----------------------------
    # Transform
    # -----------------------------
    transform = build_transform(config)

    # -----------------------------
    # Dataset paths
    # -----------------------------
    base = config["dataset"]["path"]

    train_dataset = OCRDataset(
        image_dir=os.path.join(base, config["dataset"]["train"]["images"]),
        label_path=os.path.join(base, config["dataset"]["train"]["labels"]),
        transform=transform,
    )

    val_dataset = OCRDataset(
        image_dir=os.path.join(base, config["dataset"]["val"]["images"]),
        label_path=os.path.join(base, config["dataset"]["val"]["labels"]),
        transform=transform,
    )

    print("Train samples:", len(train_dataset))
    print("Val samples:", len(val_dataset))

    # -----------------------------
    # Vocabulary
    # -----------------------------
    charset_path = config["dataset"].get("charset")
    vocab = KhmerVocab(charset_path=charset_path)

    print("Vocab size:", len(vocab))

    if hasattr(vocab, "idx_to_char"):
        print("First vocab items:", vocab.idx_to_char[:10])

    if hasattr(vocab, "char_to_idx"):
        print("Blank index:", vocab.char_to_idx.get("<blank>"))

    # -----------------------------
    # Collate function
    # -----------------------------
    collate_fn = partial(ocr_collate_fn, vocab=vocab)

    # -----------------------------
    # DataLoaders
    # -----------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config["training"].get("num_workers", 0),
        pin_memory=device.type == "cuda",
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config["training"].get("num_workers", 0),
        pin_memory=device.type == "cuda",
    )

    # -----------------------------
    # Model
    # -----------------------------
    model = build_model(config, num_classes=len(vocab)).to(device)

    # -----------------------------
    # Loss and optimizer
    # -----------------------------
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config["training"].get("scheduler_factor", 0.5),
        patience=config["training"].get("scheduler_patience", 5),
        min_lr=1e-6,
    )

    # -----------------------------
    # Resume checkpoint
    # -----------------------------
    resume_path = args.resume or config["checkpoint"].get("resume_from")

    start_epoch = 1

    if resume_path:
        start_epoch = load_checkpoint(
            resume_path,
            model,
            optimizer,
            device
        )

    # -----------------------------
    # Training setup
    # -----------------------------
    best_cer = float("inf")
    best_epoch = start_epoch
    epochs = config["training"]["epochs"]
    checkpoint_dir = config["checkpoint"]["checkpoint_dir"]

    csv_path, json_path = init_experiment_log(checkpoint_dir, config)

    # -----------------------------
    # Training loop
    # -----------------------------
    try:
        for epoch in range(start_epoch, epochs + 1):
            epoch_start = time.time()

            train_loss = train_one_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
            )

            val_cer = validate_one_epoch(
                model=model,
                dataloader=val_loader,
                vocab=vocab,
                device=device,
                blank_id=0,
            )

            epoch_time = time.time() - epoch_start
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch [{epoch}/{epochs}] "
                f"Train Loss: {train_loss:.4f} "
                f"Val CER: {val_cer:.4f} "
                f"LR: {current_lr:.2e} "
                f"Time: {epoch_time:.1f}s"
            )

            scheduler.step(val_cer)
            log_epoch(csv_path, epoch, train_loss, val_cer, current_lr, epoch_time)

            save_every = config["checkpoint"].get("save_every", 1)
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_cer=val_cer,
                checkpoint_dir=checkpoint_dir,
                save_epoch_file=(epoch % save_every == 0),
            )

            if val_cer < best_cer:
                best_cer = val_cer
                best_epoch = epoch
                save_best_model(
                    model=model,
                    checkpoint_dir=checkpoint_dir
                )
                print("Best model saved.")

        finish_experiment_log(json_path, best_cer, best_epoch)
        print(f"\nExperiment saved to: {checkpoint_dir}")

    except KeyboardInterrupt:
        finish_experiment_log(json_path, best_cer, best_epoch)
        print("\nTraining interrupted.")
        print("Last checkpoint saved in:", checkpoint_dir)


if __name__ == "__main__":
    main()