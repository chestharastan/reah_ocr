import os
import sys
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from functools import partial

# chunkmerge's own src first, traditional/src second.
# Names that exist in both (vocab, collate, train_loop, validate) resolve to
# chunkmerge's seq2seq versions; OCRDataset and utils come from traditional/.
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(1, os.path.abspath(os.path.join(_HERE, "..", "..", "traditional", "src")))

from utils import (load_config, load_checkpoint, save_checkpoint, save_best_model,
                   init_experiment_log, log_epoch, finish_experiment_log)
from dataset import OCRDataset
from vocab import KhmerVocab
from collate import ocr_collate_fn
from architectures import build_model, build_transform
from train_loop import train_one_epoch
from validate import validate_one_epoch


DEFAULT_CONFIG = os.path.join(_HERE, "..", "CBC_config", "config_chunkmerge_paper_10k.yml")


def parse_args():
    parser = argparse.ArgumentParser(description="Train chunkmerge seq2seq OCR model")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to config YAML")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    if torch.cuda.is_available() and config["training"]["device"] == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    transform = build_transform(config)

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

    charset_path = config["dataset"].get("charset")
    vocab = KhmerVocab(charset_path=charset_path)
    print("Vocab size:", len(vocab))
    print(f"Specials: pad={vocab.pad_id} bos={vocab.bos_id} eos={vocab.eos_id} unk={vocab.unk_id}")

    collate_fn = partial(ocr_collate_fn, vocab=vocab)

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

    model = build_model(
        config,
        num_classes=len(vocab),
        pad_id=vocab.pad_id,
        bos_id=vocab.bos_id,
        eos_id=vocab.eos_id,
    ).to(device)

    # Initialize LazyLinear by running a dummy forward; otherwise the optimizer
    # would be created before the lazy params materialize.
    with torch.no_grad():
        H = config["preprocessing"]["image_height"]
        W = config["preprocessing"]["image_width"]
        dummy_img = torch.zeros(1, 1, H, W, device=device)
        dummy_tgt = torch.full((1, 1), vocab.bos_id, dtype=torch.long, device=device)
        model(dummy_img, dummy_tgt)

    criterion = nn.CrossEntropyLoss(
        ignore_index=vocab.pad_id,
        label_smoothing=config["training"].get("label_smoothing", 0.0),
    )

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

    resume_path = args.resume or config["checkpoint"].get("resume_from")
    start_epoch = 1
    if resume_path:
        start_epoch = load_checkpoint(resume_path, model, optimizer, device)

    best_cer = float("inf")
    best_epoch = start_epoch
    epochs = config["training"]["epochs"]
    checkpoint_dir = config["checkpoint"]["checkpoint_dir"]

    csv_path, json_path = init_experiment_log(checkpoint_dir, config)

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
                max_len=config["model"].get("max_target_len", 256),
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
                save_best_model(model=model, checkpoint_dir=checkpoint_dir)
                print("Best model saved.")

        finish_experiment_log(json_path, best_cer, best_epoch)
        print(f"\nExperiment saved to: {checkpoint_dir}")

    except KeyboardInterrupt:
        finish_experiment_log(json_path, best_cer, best_epoch)
        print("\nTraining interrupted.")
        print("Last checkpoint saved in:", checkpoint_dir)


if __name__ == "__main__":
    main()
