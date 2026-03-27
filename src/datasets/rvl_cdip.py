from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import random

from PIL import Image, ImageFile
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# pozwala dokończyć wczytywanie uciętych plików (czasem pomaga)
ImageFile.LOAD_TRUNCATED_IMAGES = True


def build_transforms(img_size: int, train: bool) -> transforms.Compose:
    # Transforms = preprocessing + (opcjonalnie) augmentacja.
    # Dlaczego resize? Model wymaga stałego rozmiaru wejścia.
    # Dlaczego normalize ImageNet? Standard dla modeli vision, szczególnie pretrained.
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            # Delikatna losowa zmiana koloru/jasności (odporność na różne skany/druk).
            transforms.RandomApply([transforms.ColorJitter(0.1, 0.1, 0.1, 0.05)], p=0.3),
            # Delikatny obrót (np. krzywo zeskanowana kartka).
            transforms.RandomRotation(degrees=2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


class SafeImageFolderDataset(Dataset):
    """
    Dataset “ImageFolder-like”, ale z bezpiecznym ładowaniem.
    Oczekuje struktury:
      root/<split>/<class>/*.{tif,tiff,jpg,png,...}

    Pomija uszkodzone obrazy w runtime.
    (Dzięki temu trening nie pada na 1 wadliwym pliku.)
    """
    def __init__(self, split_dir: Path, transform, class_to_idx: Dict[str, int]):
        self.split_dir = split_dir
        self.transform = transform
        self.class_to_idx = class_to_idx

        # Dozwolone formaty – w praktyce RVL-CDIP u Ciebie jest w TIFF.
        exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

        # items: lista (ścieżka_do_pliku, label_int)
        items: List[Tuple[Path, int]] = []
        for cls_name, cls_idx in class_to_idx.items():
            cls_dir = split_dir / cls_name
            if not cls_dir.exists():
                continue
            for p in cls_dir.rglob("*"):
                if p.is_file() and p.suffix.lower() in exts:
                    items.append((p, cls_idx))

        if not items:
            raise RuntimeError(f"Brak plików w {split_dir}")

        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        # “Bezpieczny” getitem:
        # próbujemy kilka razy, bo czasem trafisz na uszkodzony plik.
        tries = 0
        while tries < 10:
            path, label = self.items[idx]
            try:
                # Konwertujemy do RGB, żeby mieć stały kanał (3 kanały).
                # Dokumenty mogą być grayscale / paletowe itp.
                with Image.open(path) as img:
                    img = img.convert("RGB")
                if self.transform is not None:
                    img = self.transform(img)
                return img, label
            except Exception as e:
                # Logujemy tylko przy pierwszym błędzie dla tego idx, żeby nie spamować.
                if tries == 0:
                    print(f"[WARN] skip bad image: {path} ({type(e).__name__}: {e})")
                # Losujemy inny indeks i próbujemy ponownie.
                idx = random.randint(0, len(self.items) - 1)
                tries += 1

        # Jeśli naprawdę jest dramat (np. zbyt dużo złych plików),
        # zwracamy pierwszy poprawny przykład “awaryjnie”.
        path, label = self.items[0]
        with Image.open(path) as img:
            img = img.convert("RGB")
        img = self.transform(img) if self.transform else img
        return img, label


def _discover_classes(train_dir: Path) -> Dict[str, int]:
    # Odczytujemy nazwy klas po folderach w train/.
    # Zwracamy mapowanie class_name -> idx (kolejność sortowana).
    class_names = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])
    if not class_names:
        raise RuntimeError(f"Nie znaleziono folderów klas w {train_dir}")
    return {name: i for i, name in enumerate(class_names)}


def create_dataloaders(
    data_dir: Path,
    img_size: int,
    batch_size: int,
    num_workers: int,
):
    # Tworzymy DataLoadery dla train/val/test.
    # Uwaga: pin_memory=True daje przyspieszenie transferu CPU->GPU, gdy mamy CUDA.
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    if not (train_dir.exists() and val_dir.exists() and test_dir.exists()):
        raise FileNotFoundError(
            f"Brak struktury ImageFolder. Spodziewam się folderów: train/ val/ test/ w {data_dir}"
        )

    class_to_idx = _discover_classes(train_dir)

    ds_train = SafeImageFolderDataset(train_dir, build_transforms(img_size, True), class_to_idx)
    ds_val = SafeImageFolderDataset(val_dir, build_transforms(img_size, False), class_to_idx)
    ds_test = SafeImageFolderDataset(test_dir, build_transforms(img_size, False), class_to_idx)

    loaders = {
        "train": DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True),
        "val": DataLoader(ds_val, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True),
        "test": DataLoader(ds_test, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True),
    }

    return loaders, class_to_idx