# scripts/infer.py
from pathlib import Path
import sys

# Make sure 'scripts' directory is importable when executing this file directly
scripts_dir = Path(__file__).resolve().parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

import argparse
import joblib
import numpy as np
from PIL import Image
import torch
from transformers import ViTModel, ViTImageProcessor

# Local import (preprocess.py must be in the same scripts/ folder)
from preprocess import preprocess_image

def predict(image_path, model_name, classifier_path):
    # Load processor and ViT (feature extractor)
    proc = ViTImageProcessor.from_pretrained(model_name)
    vit = ViTModel.from_pretrained(model_name).eval()

    # Load classifier
    try:
        clf_bundle = joblib.load(classifier_path)
        clf = clf_bundle['model']
        class_names = clf_bundle.get('class_names') or clf_bundle.get('classes') or clf_bundle.get('present_labels')
        # if names are indices only, try to recover
        if isinstance(class_names, (list, tuple)) and all(isinstance(x, (int, np.integer)) for x in class_names):
            # fallback mapping to generic names
            class_names = [f"class_{i}" for i in class_names]
    except Exception as e:
        print("ERROR loading classifier:", e)
        return

    img = preprocess_image(image_path, target_size=(224,224))
    if img is None:
        print("No face detected or could not read image.")
        return

    pil = Image.fromarray(img.astype('uint8'))
    inputs = proc(images=pil, return_tensors="pt")
    with torch.no_grad():
        outputs = vit(**inputs)
        feat = outputs.pooler_output.numpy()

    pred_idx = clf.predict(feat)[0]
    probs = None
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(feat)[0]

    label = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)
    print("Predicted:", label)
    if probs is not None:
        print("Probs:", dict(zip(class_names, probs.tolist())))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model_name", default="google/vit-base-patch16-224-in21k")
    parser.add_argument("--classifier", default="outputs/classifier.joblib")
    args = parser.parse_args()
    predict(args.image, args.model_name, args.classifier)
