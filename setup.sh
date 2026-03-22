#!/usr/bin/env bash
# setup.sh — Bootstrap the face recognition project
set -e

echo "🔧 Setting up Face Recognition Project..."

python3 --version || { echo "❌ Python 3.8+ required"; exit 1; }

echo "📦 Installing dependencies..."
pip install -r requirements.txt

echo "📁 Creating directories..."
mkdir -p dataset/{train,val,test}
mkdir -p models results

echo ""
echo "✅ Setup complete!"
echo ""
echo "Quick start:"
echo "  1. Capture faces from webcam:"
echo "     python scripts/collect_faces.py --name pessoa1 --count 50"
echo ""
echo "  2. Pre-process (detect & crop faces):"
echo "     python scripts/preprocess.py --input dataset/train/ --output dataset/train_processed/"
echo ""
echo "  3. Train classifier:"
echo "     python scripts/train.py --data_dir dataset/train_processed/ --epochs 30"
echo ""
echo "  4. Recognize faces:"
echo "     python scripts/recognize.py --model models/face_classifier.h5 --source foto.jpg"
echo ""
echo "  Or use the Colab notebook: notebooks/face_recognition_colab.ipynb"
