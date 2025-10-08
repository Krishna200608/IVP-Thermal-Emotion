# scripts/train_classifier.py  (fixed, robust)
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import argparse
from collections import Counter

def load_features(npz_path):
    a = np.load(npz_path, allow_pickle=True)
    X = a['features']
    y = a['labels']
    class_names = list(a['class_names'])
    return X, y, class_names

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--out_model", default="outputs/classifier.joblib")
    args = parser.parse_args()

    X, y, class_names = load_features(args.features)
    print("Loaded features:", X.shape)
    print("Loaded labels:", y.shape)
    print("Original class_names (len={}): {}".format(len(class_names), class_names))

    # Inspect distribution
    counter = Counter(y.tolist())
    print("Label distribution:", counter)

    # Determine which labels are actually present
    present_labels = sorted(list(counter.keys()))
    print("Present labels:", present_labels)

    # Build target_names filtered to present labels
    try:
        target_names_filtered = [class_names[i] for i in present_labels]
    except Exception as e:
        # Fallback: if class_names doesn't index by label (unexpected), create generic labels
        print("Warning: could not index class_names using label indices:", e)
        target_names_filtered = [f"class_{i}" for i in present_labels]

    # If too few classes to split, abort with helpful message
    if len(present_labels) < 2:
        raise SystemExit("ERROR: Need at least 2 classes with samples to train classifier. Found: {}".format(present_labels))

    # Split dataset; use stratify only if all classes have >=2 samples
    min_count = min(counter.values())
    stratify_arg = y if min_count >= 2 else None
    if stratify_arg is None:
        print("Warning: some classes have <2 samples; proceeding without stratify for train_test_split.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=stratify_arg, random_state=42)

    # Train a linear SVM
    print("Training linear SVM on embeddings...")
    svc = SVC(kernel='linear', probability=True)
    svc.fit(X_train, y_train)

    # Predict & report
    y_pred = svc.predict(X_test)
    print("Classification report (filtered to present labels):")
    print(classification_report(y_test, y_pred, labels=present_labels, target_names=target_names_filtered))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, y_pred, labels=present_labels))

    # Save model + class mapping for inference
    joblib.dump({'model': svc, 'present_labels': present_labels, 'class_names': target_names_filtered}, args.out_model)
    print("Saved model bundle to", args.out_model)
