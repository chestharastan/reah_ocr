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
from vocab import KhmerVocab, KhmerVocabAttention
from collate import ocr_collate_fn, ocr_collate_fn_attention
from architectures import build_model, build_transform
from train_loop import train_one_epoch
from train_loop_attention import train_one_epoch_attention
from validate import validate_one_epoch
from validate_attention import validate_one_epoch_attention


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


def make_loader(dataset, config, collate_fn, device, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=config["training"].get("num_workers", 0),
        pin_memory=device.type == "cuda",
    )


def main():
    args = parse_args()
    config = load_config(args.config)

    # Device
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

    test_cfg = config["dataset"].get("test")
    test_dataset = None
    if test_cfg:
        test_dataset = OCRDataset(
            image_dir=os.path.join(base, test_cfg["images"]),
            label_path=os.path.join(base, test_cfg["labels"]),
            transform=transform,
        )

    print("Train samples:", len(train_dataset))
    print("Val samples  :", len(val_dataset))
    if test_dataset:
        print("Test samples :", len(test_dataset))

    # Vocabulary + collate
    charset_path = config["dataset"].get("charset")
    arch = config["model"]["architecture"]
    decoder = config["model"].get("decoder", "ctc").lower()

    if decoder not in ("ctc", "attention"):
        raise ValueError(f"Unknown decoder '{decoder}'. Supported: 'ctc', 'attention'.")

    use_attention = decoder == "attention"

    if use_attention:
        vocab = KhmerVocabAttention(charset_path=charset_path)
        collate_fn = partial(ocr_collate_fn_attention, vocab=vocab)
    else:
        vocab = KhmerVocab(charset_path=charset_path)
        collate_fn = partial(ocr_collate_fn, vocab=vocab)

    print(f"Using {decoder} decoder (arch={arch})")
    print("Vocab size:", len(vocab))

    # DataLoaders
    train_loader = make_loader(train_dataset, config, collate_fn, device, shuffle=True)
    val_loader   = make_loader(val_dataset,   config, collate_fn, device, shuffle=False)
    test_loader  = make_loader(test_dataset,  config, collate_fn, device, shuffle=False) if test_dataset else None

    # Model
    model = build_model(config, num_classes=len(vocab)).to(device)

    criterion = None if use_attention else nn.CTCLoss(blank=0, zero_infinity=True)

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

    # Resume
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

            if use_attention:
                train_loss = train_one_epoch_attention(
                    model=model,
                    dataloader=train_loader,
                    optimizer=optimizer,
                    device=device,
                )
                val_cer, val_wer = validate_one_epoch_attention(
                    model=model,
                    dataloader=val_loader,
                    vocab=vocab,
                    device=device,
                )
            else:
                train_loss = train_one_epoch(
                    model=model,
                    dataloader=train_loader,
                    optimizer=optimizer,
                    criterion=criterion,
                    device=device,
                )
                val_cer, val_wer = validate_one_epoch(
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
                f"Time: {epoch_time:.1f}s "
                f"Loss: {train_loss:.4f} "
                f"CER: {val_cer:.4f} "
                f"WER: {val_wer:.4f}"
            )

            scheduler.step(val_cer)
            log_epoch(csv_path, epoch, epoch_time, train_loss, val_cer, val_wer)

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

        # Final test evaluation on best model
        test_cer, test_wer = None, None
        if test_loader:
            best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
            if os.path.exists(best_model_path):
                model.load_state_dict(torch.load(best_model_path, map_location=device))
                print("\nEvaluating on test set (best model)...")
                if use_attention:
                    test_cer, test_wer = validate_one_epoch_attention(
                        model=model, dataloader=test_loader, vocab=vocab, device=device
                    )
                else:
                    test_cer, test_wer = validate_one_epoch(
                        model=model, dataloader=test_loader, vocab=vocab, device=device, blank_id=0
                    )
                print(f"Test CER: {test_cer:.4f}  Test WER: {test_wer:.4f}")

        finish_experiment_log(json_path, best_cer, best_epoch, test_cer=test_cer, test_wer=test_wer)
        print(f"\nExperiment saved to: {checkpoint_dir}")

    except KeyboardInterrupt:
        finish_experiment_log(json_path, best_cer, best_epoch)
        print("\nTraining interrupted.")
        print("Last checkpoint saved in:", checkpoint_dir)


if __name__ == "__main__":
    main()
