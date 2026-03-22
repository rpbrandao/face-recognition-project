"""
recognize.py
Detect and recognize faces in images or live webcam feed.

Pipeline:
  1. MTCNN detects all face bounding boxes
  2. Each crop is classified by the trained Keras model
  3. Annotated result is displayed and/or saved

Usage — image:
    python scripts/recognize.py \
        --model  models/face_classifier.h5 \
        --source foto.jpg \
        --conf   0.35 \
        --output results/output.jpg

Usage — webcam:
    python scripts/recognize.py \
        --model  models/face_classifier.h5 \
        --source 0
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def load_model_and_classes(model_path: str):
    model = tf.keras.models.load_model(model_path)
    class_file = Path(model_path).parent / "class_names.json"
    with open(class_file) as f:
        class_indices = json.load(f)
    # Invert: index → name
    idx_to_name = {v: k for k, v in class_indices.items()}
    return model, idx_to_name


def preprocess_face(face_rgb: np.ndarray, img_size: int = 160) -> np.ndarray:
    """Resize and normalise a face crop for the classifier."""
    face = cv2.resize(face_rgb, (img_size, img_size))
    face = face.astype("float32") / 255.0
    return np.expand_dims(face, axis=0)


def draw_box(frame, x1, y1, x2, y2, label: str, conf: float, color=(255, 80, 80)):
    """Draw a bounding box with label and confidence score."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    text = f"{label} ({conf:.2f})"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)

    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, text, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


# ──────────────────────────────────────────────
# Core detection + recognition loop
# ──────────────────────────────────────────────

def process_frame(frame_bgr, detector, model, idx_to_name,
                  conf_threshold: float = 0.35, img_size: int = 160,
                  margin: int = 20):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    detections = detector.detect_faces(frame_rgb)

    for det in detections:
        if det["confidence"] < 0.9:
            continue  # skip uncertain face detections

        x, y, w, h = det["box"]
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(frame_bgr.shape[1], x + w + margin)
        y2 = min(frame_bgr.shape[0], y + h + margin)

        face_crop = frame_rgb[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue

        inp = preprocess_face(face_crop, img_size)
        preds = model.predict(inp, verbose=0)[0]
        top_idx = int(np.argmax(preds))
        top_conf = float(preds[top_idx])

        if top_conf < conf_threshold:
            label = "unknown"
        else:
            label = idx_to_name.get(top_idx, "unknown")

        draw_box(frame_bgr, x1, y1, x2, y2, label, top_conf)

    return frame_bgr


# ──────────────────────────────────────────────
# Entry points
# ──────────────────────────────────────────────

def run_image(args, detector, model, idx_to_name):
    frame = cv2.imread(args.source)
    if frame is None:
        raise FileNotFoundError(f"Cannot read image: {args.source}")

    result = process_frame(frame, detector, model, idx_to_name,
                           args.conf, args.img_size)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.output, result)
        print(f"✅ Result saved to: {args.output}")

    cv2.imshow("Face Recognition", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_webcam(args, detector, model, idx_to_name):
    cap = cv2.VideoCapture(int(args.source))
    print("📷 Webcam started — press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = process_frame(frame, detector, model, idx_to_name,
                               args.conf, args.img_size)
        cv2.imshow("Face Recognition (press q to quit)", result)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Face detection + recognition")
    parser.add_argument("--model",    required=True, help="Path to face_classifier.h5")
    parser.add_argument("--source",   required=True, help="Image path or webcam index (0)")
    parser.add_argument("--conf",     type=float, default=0.35, help="Min recognition confidence")
    parser.add_argument("--img_size", type=int,   default=160)
    parser.add_argument("--margin",   type=int,   default=20)
    parser.add_argument("--output",   default=None, help="Save result image here")
    args = parser.parse_args()

    print("Loading model...")
    model, idx_to_name = load_model_and_classes(args.model)
    print(f"  Classes: {list(idx_to_name.values())}")

    print("Loading MTCNN face detector...")
    detector = MTCNN()

    # Decide: image or video source
    try:
        cam_index = int(args.source)
        run_webcam(args, detector, model, idx_to_name)
    except ValueError:
        run_image(args, detector, model, idx_to_name)


if __name__ == "__main__":
    main()
