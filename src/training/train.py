import argparse
from pathlib import Path
from typing import Dict
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from src.utils.config import TrainConfig, set_seed, get_device
from src.utils.metrics import accuracy_top1
from src.datasets.rvl_cdip import create_dataloaders
from src.models.cnn import SimpleCNN
from src.models.vit import ViTClassifier

import csv
import json
import time
from torch.utils.tensorboard import SummaryWriter



def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler=None) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    pbar = tqdm(loader, desc="train", leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(images)
                loss = loss_fn(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()

        bs = targets.size(0)
        acc = accuracy_top1(logits.detach(), targets)
        total_loss += loss.item() * bs
        total_acc += acc * bs
        n += bs
        pbar.set_postfix(loss=loss.item(), acc=acc)

    return {"loss": total_loss / n, "acc": total_acc / n}


@torch.no_grad()
def evaluate(model, loader, loss_fn, device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    pbar = tqdm(loader, desc="val", leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = loss_fn(logits, targets)

        bs = targets.size(0)
        acc = accuracy_top1(logits, targets)
        total_loss += loss.item() * bs
        total_acc += acc * bs
        n += bs
        pbar.set_postfix(loss=loss.item(), acc=acc)

    return {"loss": total_loss / n, "acc": total_acc / n}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Folder z train/val/test")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "vit"])
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained weights (recommended for ViT)")
    parser.add_argument("--run_name", type=str, default="", help="Optional run label, e.g. lr1e-4_bs64_vit")
    args = parser.parse_args()

    cfg = TrainConfig(
        data_dir=Path(args.data_dir),
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
        amp=not args.no_amp,
    )

    set_seed(cfg.seed)
    device = get_device()

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
    run_label = f"{args.model}"
    if args.pretrained:
        run_label += "_pretrained"
    if args.run_name:
        run_label += f"_{args.run_name}"

    run_id = f"{ts}_{run_label}"

    ckpt_dir = cfg.out_dir / "checkpoints" / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best.pt"
    print(f"Run: {run_id} | checkpoints: {ckpt_dir}")
    config_path = ckpt_dir / "run_config.json"

    config_path.write_text(json.dumps({
        "data_dir": str(cfg.data_dir),
        "img_size": cfg.img_size,
        "batch_size": cfg.batch_size,
        "epochs": cfg.epochs,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "num_workers": cfg.num_workers,
        "seed": cfg.seed,
        "amp": cfg.amp,
        "model": args.model,
        "pretrained": bool(args.pretrained),
        "run_name": args.run_name,
    }, indent=2), encoding="utf-8")

    metrics_path = ckpt_dir / "metrics.csv"
    csv_file = open(metrics_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_file, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
    csv_writer.writeheader()

    tb_dir = ckpt_dir / "tb"
    writer = SummaryWriter(log_dir=str(tb_dir))

    loaders, class_to_idx = create_dataloaders(
        data_dir=cfg.data_dir,
        img_size=cfg.img_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )
    num_classes = len(class_to_idx)

    if args.model == "cnn":
        model = SimpleCNN(num_classes=num_classes).to(device)
    elif args.model == "vit":
        model = ViTClassifier(num_classes=num_classes, pretrained=args.pretrained).to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    scaler = torch.amp.GradScaler("cuda") if (cfg.amp and device.type == "cuda") else None

    best_val_acc = -1.0
    best_path = ckpt_dir / "best.pt"

    patience = 3
    no_improve = 0

    try:
        for epoch in range(1, cfg.epochs + 1):
            tr = train_one_epoch(model, loaders["train"], optimizer, loss_fn, device, scaler=scaler)
            va = evaluate(model, loaders["val"], loss_fn, device)

            print(f"Epoch {epoch:02d}/{cfg.epochs} | "
                f"train loss {tr['loss']:.4f} acc {tr['acc']:.4f} | "
                f"val loss {va['loss']:.4f} acc {va['acc']:.4f}")

            csv_writer.writerow({
                "epoch": epoch,
                "train_loss": tr["loss"],
                "train_acc": tr["acc"],
                "val_loss": va["loss"],
                "val_acc": va["acc"],
            })
            csv_file.flush()        
            writer.add_scalar("loss/train", tr["loss"], epoch)
            writer.add_scalar("loss/val", va["loss"], epoch)
            writer.add_scalar("acc/train", tr["acc"], epoch)
            writer.add_scalar("acc/val", va["acc"], epoch)
            writer.flush()

            improved = va["acc"] > best_val_acc + 1e-6  # mały próg, żeby uniknąć "szumu"

            if improved:
                best_val_acc = va["acc"]
                no_improve = 0
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "class_to_idx": class_to_idx,
                        "img_size": cfg.img_size,
                        "model_name": args.model,
                        "pretrained": bool(args.pretrained),
                    },
                    best_path,
                )
                print(f"  ✓ saved best checkpoint: {best_path} (val_acc={best_val_acc:.4f})")
            else:
                no_improve += 1
                print(f"  no improvement: {no_improve}/{patience}")

            if no_improve >= patience:
                print(f"Early stopping: no val_acc improvement for {patience} epochs.")
                break

        print(f"Done. Best val acc: {best_val_acc:.4f}")
    finally:
        writer.close()
        csv_file.close()
    
    print(f"Saved metrics: {metrics_path}")
    print(f"TensorBoard logs: {tb_dir}")
    print(f"Best checkpoint: {best_path}")
if __name__ == "__main__":
    main()