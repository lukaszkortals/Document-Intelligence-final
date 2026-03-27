import argparse
import shutil
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageFile

# Ustawienie pozwalające Pillow spróbować dokończyć ładowanie uciętych plików.
# Nie naprawi totalnie uszkodzonych plików, ale czasem ratuje “prawie dobre” TIFF-y.
ImageFile.LOAD_TRUNCATED_IMAGES = True


def iter_image_files(root: Path, exts: Tuple[str, ...]) -> List[Path]:
    # Zbiera wszystkie pliki z dozwolonymi rozszerzeniami (rekurencyjnie).
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return files


def is_image_readable(path: Path) -> bool:
    # Sprawdza, czy obraz da się otworzyć i zdekodować.
    # verify() sprawdza integralność, ale czasem nie dekoduje wszystkiego,
    # więc robimy też drugi open + convert("RGB") dla pewności.
    try:
        with Image.open(path) as img:
            img.verify()  # szybka walidacja pliku
        # verify() nie zawsze dekoduje w pełni – dobijmy pewnością:
        with Image.open(path) as img:
            img.convert("RGB")
        return True
    except Exception:
        return False


def main():
    # Ten skrypt służy do offline’owej walidacji datasetu:
    # - wykrywa pliki, których Pillow nie potrafi wczytać
    # - zapisuje listę “bad images”
    # - opcjonalnie przenosi uszkodzone pliki do osobnego katalog
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True,
                        help="Folder datasetu (np. data/processed/RVL-CDIP_subset)")
    parser.add_argument("--ext", type=str, default="tif,tiff,jpg,jpeg,png,bmp",
                        help="Rozszerzenia plików (comma-separated)")
    parser.add_argument("--move_bad_to", type=str, default="",
                        help="Jeśli ustawione: przenieś złe pliki do tego folderu")
    parser.add_argument("--limit", type=int, default=0,
                        help="Dla testu: sprawdź tylko N plików (0 = wszystkie)")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Nie ma folderu: {root}")

    exts = tuple(("." + e.strip().lower()) for e in args.ext.split(","))
    files = iter_image_files(root, exts)

    # Ułatwia szybkie testy bez skanowania całego datasetu.
    if args.limit and args.limit > 0:
        files = files[:args.limit]

    print(f"Sprawdzam plików: {len(files)} w {root}")
    bad: List[Path] = []

    for i, f in enumerate(files, start=1):
        ok = is_image_readable(f)
        if not ok:
            bad.append(f)
        # Progres co 1000 plików, żeby było widać, że skrypt żyje.
        if i % 1000 == 0:
            print(f"  progress: {i}/{len(files)} | bad: {len(bad)}")

    print("\nDONE")
    print(f"Bad files: {len(bad)}")

    # Zapis listy uszkodzonych plików do outputs/bad_images.txt
    out_report = Path("outputs") / "bad_images.txt"
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text("\n".join(str(p) for p in bad), encoding="utf-8")
    print(f"Zapisano listę: {out_report}")

    # Opcjonalne przenoszenie “bad files” do osobnego katalogu.
    # Przydatne, gdy trening się wywala na pojedynczym pliku.
    if args.move_bad_to:
        dst_root = Path(args.move_bad_to)
        dst_root.mkdir(parents=True, exist_ok=True)

        moved = 0
        for p in bad:
            # zachowujemy strukturę względną względem root
            rel = p.relative_to(root)
            dst = dst_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.move(str(p), str(dst))
                moved += 1
            except Exception as e:
                print(f"[WARN] nie mogę przenieść {p} -> {dst}: {e}")

        print(f"Przeniesiono: {moved} złych plików do: {dst_root}")


if __name__ == "__main__":
    main()