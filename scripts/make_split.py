import argparse
import os
import random
import shutil
from pathlib import Path


def safe_link_or_copy(src: Path, dst: Path, use_hardlinks: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return

    if use_hardlinks:
        try:
            os.link(src, dst)  # hardlink (NTFS)
            return
        except OSError:
            pass  # fallback do copy

    shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True,
                        help="Folder z klasami: src/<class>/*.jpg")
    parser.add_argument("--dst", type=str, required=True,
                        help="Folder wyjściowy: dst/{train,val,test}/<class>/*.jpg")

    parser.add_argument("--train", type=int, default=800, help="Ile plików na klasę do train")
    parser.add_argument("--val", type=int, default=100, help="Ile plików na klasę do val")
    parser.add_argument("--test", type=int, default=100, help="Ile plików na klasę do test")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hardlinks", action="store_true",
                        help="Użyj hardlinków zamiast kopiowania (szybko i bez zajmowania miejsca)")

    parser.add_argument("--ext", type=str, default="jpg,png,jpeg,tif,tiff",
                        help="Dozwolone rozszerzenia (comma-separated)")

    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    exts = {("." + e.strip().lower()) for e in args.ext.split(",")}

    if not src.exists():
        raise FileNotFoundError(f"Nie ma src: {src}")

    random.seed(args.seed)

    class_dirs = [p for p in src.iterdir() if p.is_dir()]
    if not class_dirs:
        raise RuntimeError(f"Nie znaleziono folderów klas w: {src}")

    print(f"Znaleziono klas: {len(class_dirs)}")
    print(f"Tworzę subsety per klasa: train={args.train}, val={args.val}, test={args.test}")
    print(f"Tryb: {'hardlinks' if args.hardlinks else 'copy'}")

    for cdir in class_dirs:
        cls = cdir.name
        files = [p for p in cdir.rglob("*") if p.is_file() and p.suffix.lower() in exts]

        if len(files) < (args.train + args.val + args.test):
            print(f"[WARN] Klasa '{cls}' ma tylko {len(files)} plików — za mało na żądane subsety. "
                  f"Zrobię ile się da proporcjonalnie.")
        random.shuffle(files)

        # bierzemy ile się da
        n_train = min(args.train, len(files))
        n_val = min(args.val, max(0, len(files) - n_train))
        n_test = min(args.test, max(0, len(files) - n_train - n_val))

        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:n_train + n_val + n_test]

        for split_name, split_files in [("train", train_files), ("val", val_files), ("test", test_files)]:
            out_dir = dst / split_name / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            for f in split_files:
                safe_link_or_copy(f, out_dir / f.name, use_hardlinks=args.hardlinks)

        print(f"{cls}: train={len(train_files)} val={len(val_files)} test={len(test_files)}")

    print("\nDONE.")
    print(f"Wynik: {dst}")
    print("Teraz możesz trenować: python -m src.training.train --data_dir <dst>")


if __name__ == "__main__":
    main()