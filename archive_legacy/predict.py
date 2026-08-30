import onnxruntime as ort
import numpy as np
from PIL import Image
import pandas as pd
from pathlib import Path
import sys

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    
    img_data = np.array(img, dtype=np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_data = (img_data - mean) / std

    img_data = np.transpose(img_data, (2, 0, 1))
    img_data = np.expand_dims(img_data, axis=0).astype(np.float32)
    return img_data

def predict(image_path, species="fish", backbone="efficientnet_b0"):
    onnx_path = Path(f"exports/{backbone}_{species}_optimized.onnx")
    csv_path = Path(f"datasets/processed/{species}_processed_manifest.csv")

    if not onnx_path.exists():
        print(f"[ERROR] ONNX model not found: {onnx_path}")
        return

    df = pd.read_csv(csv_path)
    classes = sorted(df[df['fold'] >= 0]['label'].unique())

    session = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name
    
    img_tensor = preprocess_image(image_path)
    
    logits = session.run(None, {input_name: img_tensor})[0][0]
    
    exp_preds = np.exp(logits - np.max(logits))
    probs = exp_preds / exp_preds.sum()
    
    top_idx = np.argmax(probs)
    top_class = classes[top_idx]
    top_conf = probs[top_idx] * 100

    print("\n" + "="*40)
    print("🔬 AQUAVISION INFERENCE ENGINE 🔬")
    print("="*40)
    print(f"File:       {Path(image_path).name}")
    print(f"Prediction: {top_class}")
    print(f"Confidence: {top_conf:.2f}%")
    print("="*40 + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
    else:
        predict(sys.argv[1])
