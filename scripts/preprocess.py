"""
preprocess.py
Detects and crops faces from a dataset folder using MTCNN,
saving aligned face images ready for classifier training.

Usage:
    python scripts/preprocess.py \
        --input  dataset/train/ \
        --output dataset/train_processed/ \
        --size   160 \
        --margin 20
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from mtcnn import MTCNN
from tqdm import tqdm


def extract_face(detector: MTCNN, img_rgb: np.ndarray,
                 target_size: int = 160, margin: int = 20) -> np.ndarray | None:
    """Detect and crop the largest face in the image."""
    results = detector.detect_faces(img_rgb)
    if not results:
        return None

    # Pick the face with highest confidence
    best = max(results, key=lambda r: r["confidence"])
    x, y, w, h = best["box"]

    # Apply margin
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(img_rgb.shape[1], x + w + margin)
    y2 = min(img_rgb.shape[0], y + h + margin)

    face = img_rgb[y1:y2, x1:x2]
    face = Image.fromarray(face).resize((target_size, target_size))
    return np.array(face)


def process_dataset(input_dir: Path, output_dir: Path,
                    target_size: int = 160, margin: int = 20):
    detector = MTCNN()
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    class_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    print(f"Found {len(class_dirs)} classes in {input_dir}")

    total_ok = total_skip = 0

    for class_dir in sorted(class_dirs):
        out_class = output_dir / class_dir.name
        out_class.mkdir(parents=True, exist_ok=True)

        images = [f for f in class_dir.iterdir() if f.suffix.lower() in IMG_EXTS]
        print(f"\n  [{class_dir.name}] {len(images)} images")

        for img_path in tqdm(images, desc=f"  {class_dir.name}", leave=False):
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                total_skip += 1
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            face = extract_face(detector, img_rgb, target_size, margin)

            if face is None:
                print(f"    [NO FACE] {img_path.name}")
                total_skip += 1
                continue

            out_path = out_class / img_path.name
            Image.fromarray(face).save(str(out_path))
            total_ok += 1

    print(f"\n✅ Done!  Processed: {total_ok}  |  Skipped: {total_skip}")
    print(f"   Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Crop aligned faces for training")
    parser.add_argument("--input",  required=True, help="Input dataset root (class subdirs)")
    parser.add_argument("--output", required=True, help="Output directory for cropped faces")
    parser.add_argument("--size",   type=int, default=160, help="Face crop size (px)")
    parser.add_argument("--margin", type=int, default=20,  help="Extra margin around bbox (px)")
    args = parser.parse_args()

    process_dataset(
        Path(args.input), Path(args.output),
        target_size=args.size, margin=args.margin
    )


if __name__ == "__main__":
    main()
