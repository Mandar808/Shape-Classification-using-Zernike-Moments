import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from joblib import dump
from pyfeats import zernikes_moments

# Paths
dataset_path = "C:/Users/91741/Downloads/pyfeats/capital_alphabet_dataset"

# Zernike parameter
zernike_radius = 64  # Image must be square, so 128x128 image gives radius 64

# Data storage
features = []
labels = []

# Load and process dataset
print("[INFO] Loading dataset and extracting Zernike features...")
for label in sorted(os.listdir(dataset_path)):
    folder_path = os.path.join(dataset_path, label)
    if not os.path.isdir(folder_path):
        continue
    for filename in tqdm(os.listdir(folder_path), desc=f"Processing {label}"):
        image_path = os.path.join(folder_path, filename)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (128, 128))
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Extract Zernike moments (remove 'degree' param)
        zernike_feats, _ = zernikes_moments(binary, radius=zernike_radius)
        features.append(zernike_feats)
        labels.append(label)

# Encode labels
le = LabelEncoder()
encoded_labels = le.fit_transform(labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    features, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
)

# Train SVM
print("[INFO] Training SVM classifier...")
clf = SVC(kernel="rbf", probability=True)
clf.fit(X_train, y_train)

# Evaluate
print("[INFO] Evaluating classifier...")
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save model
dump(clf, "zernike_alphabet_svm_model.joblib")
dump(le, "alphabet_label_encoder.joblib")
print("✅ Model and label encoder saved successfully!")
