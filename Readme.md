# Thermal Emotion Recognition Project 🔥📸🧠

This project uses **ViT embeddings + SVM** to detect emotions from thermal images. It includes preprocessing, feature extraction, classifier training, and batch inference. 🌡️🤖😊

---

## **Project Structure** 🗂️📁✨

```
Code/
├── scripts/
│   ├── preprocess.py
│   ├── extract_features.py
│   ├── train_classifier.py
│   ├── infer.py
│   ├── batch_infer.py
│   └── preprocess_image.py
├── data/               # Input thermal images
├── outputs/            # Generated processed images, features, models, predictions
├── README.md
└── requirements.txt
```

---

## **Setup Environment** ⚙️💻🔧

```bash
# Create a virtual environment using Python 3.11
python3.11 -m venv thermal_env

# Activate the environment
# 🪟 On Windows (PowerShell)
.\thermal_env\Scripts\Activate.ps1

# 🧠 OR on Command Prompt (cmd)
thermal_env\Scripts\activate.bat

# 🐧 On Linux / macOS
source thermal_env/bin/activate

# Upgrade pip (recommended)
python -m pip install --upgrade pip

# Install all required dependencies
pip install -r requirements.txt
```

**requirements.txt** should include: 📜🧩💡

```
torch
transformers
numpy
scikit-learn
opencv-python
Pillow
joblib
```

---

## **Step-by-Step Usage** 🚀🧠📊

### 1. Preprocess Images

```bash
python scripts/preprocess.py \
    --data_dir data \
    --out_images_dir outputs/processed \
    --target_size 224 224 \
    --save_npz outputs/preprocessed.npz
```

### 2. Extract Features (ViT Embeddings)

```bash
python scripts/extract_features.py \
    --npz_input outputs/preprocessed.npz \
    --out_npz outputs/features.npz \
    --batch_size 8 \
    --num_threads 6
```

### 3. Train Classifier (SVM)

```bash
python scripts/train_classifier.py \
    --features outputs/features.npz \
    --out_model outputs/classifier.joblib
```

### 4. Test Single Image

```bash
python scripts/infer.py \
    --image data/Happy/example.jpg
```

### 5. Batch Inference

```bash
python -m scripts.batch_infer \
    --data_dir data \
    --classifier outputs/classifier.joblib \
    --out_csv outputs/predictions.csv
```

**Output:** `predictions.csv` containing predicted labels and probabilities. 📊📁✅

---

## **6. Performance Notes** ⏱️⚡📈

Use Python’s `time` module to measure average inference time per image. ⌛🐍🖥️

Example:

```bash
python -m scripts.batch_infer --data_dir data --classifier outputs/classifier.joblib --out_csv outputs/predictions.csv
```

At the end, the script prints average, min, and max inference times. 🧮📉📊

**Hardware info** can be displayed via: 💻🧰⚙️

```bash
python -c "import torch; print(torch.__version__); print(torch.get_num_threads())"
python -c "import platform; print(platform.platform())"
```

---

## **7. Optional Improvements** 🧪📈🔍

* Compare SVM on ViT features vs fine-tuning ViT head for classifier improvement.
* Add more images or augment dataset to increase accuracy. 🎯📸💪

---

## **Author** 👨‍💻🎓💡

**IIT2023139 — Krishna Sikheriya**
