"""
collect_faces.py
Capture face crops from a webcam or video to build a training dataset.

Usage — webcam (interactive):
    python scripts/collect_faces.py \
        --name   sheldon \
        --output dataset/train/ \
        --count  50

Usage — from video file:
    python scripts/collect_faces.py \
        --name    sheldon \
        --source  video.mp4 \
        --output  dataset/train/ \
        --count   100 \
        --step    5
"""

import argparse
import time
from pathlib import Path

import cv2
from mtcnn import MTCNN


def collect(name: str, source, output_dir: Path,
            count: int = 50, step: int = 1, size: int = 160):

    out_person = output_dir / name
    out_person.mkdir(parents=True, exist_ok=True)

    detector = MTCNN()
    cap = cv2.VideoCapture(source)
    saved = frame_idx = 0

    print(f"📸 Collecting {count} face images for '{name}'")
    print("   Press 'q' to stop early.\n")

    while saved < count:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % step != 0:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(rgb)

        if not results:
            continue

        best = max(results, key=lambda r: r["confidence"])
        if best["confidence"] < 0.9:
            continue

        x, y, w, h = best["box"]
        margin = 20
        x1, y1 = max(0, x - margin), max(0, y - margin)
        x2, y2 = min(frame.shape[1], x + w + margin), min(frame.shape[0], y + h + margin)

        face = rgb[y1:y2, x1:x2]
        face_bgr = cv2.cvtColor(cv2.resize(face, (size, size)), cv2.COLOR_RGB2BGR)

        filename = out_person / f"{name}_{saved:04d}.jpg"
        cv2.imwrite(str(filename), face_bgr)
        saved += 1

        # Show progress on frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Saved {saved}/{count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Collecting faces (press q to stop)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Saved {saved} images to: {out_person}")


def main():
    parser = argparse.ArgumentParser(description="Collect face images for training")
    parser.add_argument("--name",   required=True, help="Person name (subfolder)")
    parser.add_argument("--source", default=0,     help="Webcam index or video path")
    parser.add_argument("--output", default="dataset/train/")
    parser.add_argument("--count",  type=int, default=50,  help="How many images to capture")
    parser.add_argument("--step",   type=int, default=1,   help="Capture every Nth frame")
    parser.add_argument("--size",   type=int, default=160, help="Output crop size")
    args = parser.parse_args()

    # Try to parse source as int (webcam index)
    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    collect(args.name, source, Path(args.output),
            count=args.count, step=args.step, size=args.size)


if __name__ == "__main__":
    main()
