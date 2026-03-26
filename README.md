## Stage 1 — Document Image Classification (SimpleCNN baseline)

### Goal
Classify document type from **image only** (no OCR / no text).

### Data
RVL-CDIP subset (16 classes) split into train/val/test using `scripts/make_split.py`.  
Images are TIFFs; corrupted files can be detected with `scripts/validate_images.py` (and are also safely skipped at load time).

### Model
**SimpleCNN** baseline in PyTorch (scratch):
- 4 convolutional blocks:
  - Conv(3→32) + ReLU + MaxPool
  - Conv(32→64) + ReLU + MaxPool
  - Conv(64→128) + ReLU + MaxPool
  - Conv(128→256) + ReLU
- Global Average Pooling: `AdaptiveAvgPool2d((1, 1))`
- MLP classifier: `Linear(256→256) + ReLU + Dropout(0.2) + Linear(256→num_classes)`

---

## Stage 2 — Vision Transformer (ViT) fine-tuning

### Model
**ViT-B/16** from `torchvision`, **pretrained** (ImageNet) + fine-tuned on RVL-CDIP subset:
- backbone: `vit_b_16(weights=ViT_B_16_Weights.DEFAULT)`
- classification head replaced to `num_classes`

---

## How to run

### (Optional) Validate dataset images
Scans the dataset and reports unreadable images. Optionally moves them to a separate folder.


Validate images:
```bash
python scripts/validate_images.py --root data/processed/RVL-CDIP_subset --move_bad_to data/bad_images
```
Create subset:
```bash
python scripts/make_split.py --src data/raw/RVL-CDIP --dst data/processed/RVL-CDIP_subset --train 800 --val 100 --test 100 --hardlinks
```
Train:
```bash
python -m src.training.train --data_dir data/processed/RVL-CDIP_subset --model cnn --epochs 30 --batch_size 64 --img_size 224 --num_workers 8
python -m src.training.train --data_dir data/processed/RVL-CDIP_subset --model vit --pretrained --epochs 20 --batch_size 64 --num_workers 8 --lr 0.0001 --run_name lr1e-4_bs64
```
Evaluate:
```bash
python -m src.training.evaluate --data_dir data/processed/RVL-CDIP_subset --ckpt outputs/checkpoints/20260119-213309/best.pt --batch_size 64 --num_workers 8
python -m src.training.evaluate --data_dir data/processed/RVL-CDIP_subset --ckpt outputs/checkpoints/20260305-221525-161_vit_pretrained_lr1e-4_bs64/best.pt --batch_size 64 --num_workers 8
```
TensonBoard:
```bash
python -m tensorboard.main --logdir outputs\checkpoints --port 6006
```

## Results (RVL-CDIP subset)

| Model | Pretrained | Img | Batch | LR | Epochs | Test acc | Test macro F1 | Run artifacts |
|------|------------|-----|-------|----|--------|----------|---------------|--------------|
| SimpleCNN (scratch) | no | 224 | 64 | 1e-3 | 20 | 0.6319 | 0.6283 | `outputs/checkpoints/20260119-213309/` |
| ViT-B/16 | yes | 224 | 64 | 1e-4 | early stop (<=30) | 0.7937 | 0.7934 | `outputs/checkpoints/20260305-221525-161_vit_pretrained_lr1e-4_bs64/` |

### Takeaways
- Pretrained ViT-B/16 outperformed the scratch CNN baseline by ~12pp test accuracy on the same dataset split.
- ViT improved performance across all classes, including previously difficult ones (e.g., scientific_report).

