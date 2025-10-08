# scripts/batch_infer.py
import argparse
from pathlib import Path
import joblib
import numpy as np
from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import torch
from scripts.preprocess import preprocess_image
import csv
import time


def predict_image(img_path, vit, proc, clf, class_names):
    img = preprocess_image(img_path, target_size=(224,224))
    if img is None:
        return None, None
    pil = Image.fromarray(img.astype('uint8'))
    inputs = proc(images=pil, return_tensors="pt")
    with torch.no_grad():
        outputs = vit(**inputs)
        feat = outputs.pooler_output.numpy()
    pred_idx = int(clf.predict(feat)[0])
    probs = clf.predict_proba(feat)[0] if hasattr(clf, "predict_proba") else None
    label = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)
    return label, probs


def main(args):
    data_dir = Path(args.data_dir)
    out_csv = Path(args.out_csv)
    model_name = args.model_name
    classifier_path = args.classifier

    # load vit & processor once
    print("Loading ViT and processor (CPU)...")
    proc = ViTImageProcessor.from_pretrained(model_name)
    vit = ViTModel.from_pretrained(model_name).eval()

    print("Loading classifier...")
    bundle = joblib.load(classifier_path)
    clf = bundle['model']
    class_names = bundle.get('class_names') or bundle.get('classes') or bundle.get('present_labels')
    if isinstance(class_names, (list,tuple)) and all(isinstance(x,(int,np.integer)) for x in class_names):
        class_names = [f"class_{i}" for i in class_names]

    img_paths = sorted([p for p in data_dir.rglob("*") if p.suffix.lower() in ('.jpg','.jpeg','.png')])
    print(f"Found {len(img_paths)} images")

    times = []
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path','pred_label','pred_probabilities'])
        for p in img_paths:
            t0 = time.time()
            label, probs = predict_image(p, vit, proc, clf, class_names)
            t1 = time.time()
            times.append(t1 - t0)

            if label is None:
                writer.writerow([str(p), 'NO_FACE', ''])
            else:
                probs_str = ''
                if probs is not None:
                    probs_str = ';'.join([f"{class_names[i]}:{probs[i]:.4f}" for i in range(len(class_names))])
                writer.writerow([str(p), label, probs_str])

    print("Saved results to", out_csv)
    total_images = len(times)
    total_time = sum(times)
    print(f"Processed {total_images} images in {total_time:.2f} seconds")
    print(f"Average inference time per image: {total_time/total_images:.3f} seconds")
    print(f"Max per image: {max(times):.3f}, Min per image: {min(times):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="root folder with class subfolders or images")
    parser.add_argument("--classifier", default="outputs/classifier.joblib")
    parser.add_argument("--model_name", default="google/vit-base-patch16-224-in21k")
    parser.add_argument("--out_csv", default="outputs/predictions.csv")
    args = parser.parse_args()
    main(args)
