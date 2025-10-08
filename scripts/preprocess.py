# scripts/preprocess.py
import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import cv2.data

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_image(image_path, target_size=(224,224)):
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.02, minNeighbors=2)
    if len(faces) == 0:
        return None
    # choose largest face
    x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
    face = gray[y:y+h, x:x+w]
    # resize to model expected size (ViT uses 224x224)
    face_resized = cv2.resize(face, target_size, interpolation=cv2.INTER_AREA)
    # convert to 3 channels
    face_3 = cv2.cvtColor(face_resized, cv2.COLOR_GRAY2RGB)
    return face_3

def build_dataset(data_dir, out_images_dir=None, target_size=(224,224)):
    data_dir = Path(data_dir)
    images = []
    labels = []
    class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    for idx, cls in enumerate(class_names):
        cls_dir = data_dir / cls
        for p in tqdm(list(cls_dir.glob("*.*")), desc=f"Processing {cls}"):
            if p.suffix.lower() not in {'.jpg','.jpeg','.png','.bmp'}:
                continue
            proc = preprocess_image(p, target_size)
            if proc is not None:
                images.append(proc)
                labels.append(idx)
                if out_images_dir:
                    # optionally save processed images for inspection
                    out_path = Path(out_images_dir) / cls
                    out_path.mkdir(parents=True, exist_ok=True)
                    fn = out_path / p.name
                    cv2.imwrite(str(fn), cv2.cvtColor(proc, cv2.COLOR_RGB2BGR))
    images = np.stack(images) if images else np.zeros((0, *target_size, 3), dtype=np.uint8)
    labels = np.array(labels, dtype=np.int64)
    return images, labels, class_names

if __name__ == "__main__":
    import argparse, numpy as np
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_images_dir", default=None)   # optional
    parser.add_argument("--target_size", type=int, nargs=2, default=(224,224))
    parser.add_argument("--save_npz", default="outputs/preprocessed.npz")
    args = parser.parse_args()
    images, labels, class_names = build_dataset(args.data_dir, args.out_images_dir, tuple(args.target_size))
    print("Processed images:", images.shape, "labels:", labels.shape)
    np.savez_compressed(args.save_npz, images=images, labels=labels, class_names=class_names)
    print("Saved to", args.save_npz)
