# This script is for cutting the image names, we use too many digits now and can get rounding issues?

from pathlib import Path
import shutil
import argparse


def cut_id(filename_stem: str, digits_to_cut: int) -> str:
    """
    Cuts digits from the end of a numeric filename stem.
    Example:
    1724169343107751600 -> 17241693431077 when digits_to_cut=5

    Usage:

    python copy_images_cut_ids.py \
        /cluster/home/henrban/aquaculture-perception/data-processing/vision/SOLAQUA/raw_processed/all_images/2024-08-20_17-55-40 \
        /cluster/home/henrban/aquaculture-perception/data-processing/vision/SOLAQUA/raw_processed/cut_images/2024-08-20_17-55-40 \
        --digits 5
    """
    if not filename_stem.isdigit():
        raise ValueError(f"Filename stem is not numeric: {filename_stem}")

    if len(filename_stem) <= digits_to_cut:
        raise ValueError(f"Cannot cut {digits_to_cut} digits from {filename_stem}")

    return filename_stem[:-digits_to_cut]


def copy_and_rename_images(source_dir: Path, output_dir: Path, digits_to_cut: int = 5):
    if not source_dir.exists():
        raise FileNotFoundError(f"Source folder does not exist: {source_dir}")

    if output_dir.exists():
        raise FileExistsError(
            f"Output folder already exists: {output_dir}\n"
            "Delete it or choose another output folder to avoid overwriting files."
        )

    output_dir.mkdir(parents=True)

    copied = 0
    skipped = 0
    seen_outputs = set()

    for image_path in source_dir.rglob("*"):
        if image_path.suffix.lower() not in [".jpg", ".jpeg"]:
            continue

        relative_parent = image_path.parent.relative_to(source_dir)
        target_parent = output_dir / relative_parent
        target_parent.mkdir(parents=True, exist_ok=True)

        try:
            new_stem = cut_id(image_path.stem, digits_to_cut)
        except ValueError as e:
            print(f"Skipping {image_path}: {e}")
            skipped += 1
            continue

        new_name = new_stem + image_path.suffix.lower()
        target_path = target_parent / new_name

        if target_path in seen_outputs or target_path.exists():
            raise RuntimeError(
                f"ID collision detected:\n"
                f"Original file: {image_path}\n"
                f"New file: {target_path}\n"
                "Cutting this many digits would create duplicate filenames."
            )

        shutil.copy2(image_path, target_path)
        seen_outputs.add(target_path)
        copied += 1

    print(f"Done.")
    print(f"Copied images: {copied}")
    print(f"Skipped images: {skipped}")
    print(f"Output folder: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Copy JPG images to a new folder and cut digits from timestamp filenames."
    )

    parser.add_argument(
        "source_dir",
        type=Path,
        help="Folder containing the original images."
    )

    parser.add_argument(
        "output_dir",
        type=Path,
        help="New folder where copied/renamed images will be written."
    )

    parser.add_argument(
        "--digits",
        type=int,
        default=5,
        help="Number of digits to cut from the end of each image ID. Default: 5."
    )

    args = parser.parse_args()

    copy_and_rename_images(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        digits_to_cut=args.digits
    )