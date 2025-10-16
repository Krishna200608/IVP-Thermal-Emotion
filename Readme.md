# **Thermal Emotion Recognition Project** 🔥📸🧠

This project uses **Vision Transformer (ViT) embeddings + SVM** to detect emotions from thermal facial images.
It includes **preprocessing**, **feature extraction**, **classifier training**, and **batch inference**. 🌡️🤖😊

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
├── outputs/            # Processed images, features, models, predictions
├── README.md
└── requirements.txt
```

---

## **Setup Environment** ⚙️💻🔧

```powershell
# ✅ Create a virtual environment using Python 3.11
py -3.11 -m venv thermal_env311

# 🔹 Activate the environment in PowerShell
.\thermal_env311\Scripts\activate

# 🧠 Alternatively, activate in Command Prompt (cmd)
thermal_env311\Scripts\activate.bat

# ⬆️ Upgrade pip
python -m pip install --upgrade pip

# 📦 Install dependencies
pip install -r requirements.txt
```

**requirements.txt** should include:

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

### **1. Preprocess Images**

```powershell
python scripts\preprocess.py `
    --data_dir data `
    --out_images_dir outputs\processed `
    --target_size 224 224 `
    --save_npz outputs\preprocessed.npz
```

---

### **2. Extract Features (ViT Embeddings)**

```powershell
python scripts\extract_features.py `
    --npz_input outputs\preprocessed.npz `
    --out_npz outputs\features.npz `
    --batch_size 8 `
    --num_threads 6
```

---

### **3. Train Classifier (SVM)**

```powershell
python scripts\train_classifier.py `
    --features outputs\features.npz `
    --out_model outputs\classifier.joblib
```

---

### **4. Test Single Image**

```powershell
python scripts\infer.py `
    --image data\Happy\0.jpg
```

---

### **5. Batch Inference**

```powershell
python -m scripts.batch_infer `
    --data_dir data `
    --classifier outputs\classifier.joblib `
    --out_csv outputs\predictions.csv
```

**Output:**
A file named `predictions.csv` containing **predicted emotion labels and probabilities**. 📊✅

---

## **6. Performance Notes** ⏱️⚡📈

Measure average inference time per image:

```powershell
python -m scripts.batch_infer `
    --data_dir data `
    --classifier outputs\classifier.joblib `
    --out_csv outputs\predictions.csv
```

Get hardware details:

```powershell
python -c "import torch; print(torch.__version__); print(torch.get_num_threads())"
python -c "import platform; print(platform.platform())"
```

---

## **7. Optional Improvements** 🧪📈🔍

* Compare **SVM on ViT embeddings** vs **fine-tuned ViT head** for performance improvement.
* Add **data augmentation** to improve model robustness.
* Experiment with **different kernels (RBF, polynomial)** in SVM.

---

## **Author** 👨‍💻🎓💡

**IIT2023139 — Krishna Sikheriya**

---
