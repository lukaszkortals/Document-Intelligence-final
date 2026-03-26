import argparse
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

from src.utils.config import get_device
from src.datasets.rvl_cdip import create_dataloaders
from src.models.vit import ViTClassifier
from src.models.cnn import SimpleCNN



@torch.no_grad()
def run_test(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []

    for images, targets in tqdm(loader, desc="test"):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        preds = torch.argmax(logits, dim=1)

        y_true.extend(targets.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    return y_true, y_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--out_dir", type=str, default="",
                    help="Folder wyjściowy (np. outputs/checkpoints/<run_id>). Jeśli pusty, użyje folderu checkpointu.")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    out_dir = Path(args.out_dir) if args.out_dir else ckpt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model_name = ckpt.get("model_name", "cnn")
    pretrained = ckpt.get("pretrained", False)
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)
    img_size = ckpt.get("img_size", 224)

    loaders, _ = create_dataloaders(
        data_dir=Path(args.data_dir),
        img_size=img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    if model_name == "cnn":
        model = SimpleCNN(num_classes=num_classes).to(device)
    elif model_name == "vit":
        model = ViTClassifier(num_classes=num_classes, pretrained=pretrained).to(device)
    else:
        raise ValueError(f"Unknown model_name in checkpoint: {model_name}")

    model.load_state_dict(ckpt["model_state"])

    y_true, y_pred = run_test(model, loaders["test"], device)

    print("\nClassification report:")
    target_names = [idx_to_class[i] for i in range(num_classes)]
    report_text = classification_report(y_true, y_pred, target_names=target_names, digits=4)
    print(report_text)

    out_txt = out_dir / "classification_report.txt"
    out_txt.write_text(report_text, encoding="utf-8")
    print(f"Saved classification report to: {out_txt}")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 10))
    plt.imshow(cm)
    plt.title("Confusion matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(num_classes), target_names, rotation=90)
    plt.yticks(range(num_classes), target_names)
    plt.tight_layout()
    out_path = out_dir / "confusion_matrix.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    print(f"\nSaved confusion matrix to: {out_path}")


if __name__ == "__main__":
    main()