import cv2
import numpy as np
import mahotas
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Step 1: Draw basic shapes (circle, square, triangle)
def draw_shape(shape, size=128):
    img = np.zeros((size, size), dtype=np.uint8)
    if shape == 'circle':
        cv2.circle(img, (size//2, size//2), size//3, 255, -1)
    elif shape == 'square':
        offset = size//4
        cv2.rectangle(img, (offset, offset), (size-offset, size-offset), 255, -1)
    elif shape == 'triangle':
        pts = np.array([[size//2, size//4], [size//4, 3*size//4], [3*size//4, 3*size//4]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.drawContours(img, [pts], 0, 255, -1)
    return img

# Step 2: Extract Zernike features
def extract_zernike_features(image, radius=64, degree=8):
    image = image.astype(np.uint8)
    return mahotas.features.zernike_moments(image, radius=radius, degree=degree)

# Step 3: Generate dataset
def generate_dataset(num_images=100, img_size=128):
    data = []
    labels = []
    shapes = ['circle', 'triangle', 'square']
    for shape in shapes:
        for _ in range(num_images):
            img = draw_shape(shape, img_size)
            features = extract_zernike_features(img, radius=img_size//2)
            data.append(features)
            labels.append(shape)
    return np.array(data), np.array(labels)

# Step 4: Train a model
def train_model(X, y):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    model = SVC(kernel='linear', C=1.0)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
    
    return model, le, X_test, y_test, y_pred

# Step 5: Visualize predictions
def visualize_predictions(X_test, y_test, y_pred, label_encoder):
    shapes = ['circle', 'triangle', 'square']
    plt.figure(figsize=(12, 6))
    for i in range(6):
        img = draw_shape(label_encoder.inverse_transform([y_test[i]])[0])
        plt.subplot(2, 3, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(f"Predicted: {label_encoder.inverse_transform([y_pred[i]])[0]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Run the entire pipeline
if __name__ == "__main__":
    print("[INFO] Generating dataset...")
    X, y = generate_dataset(num_images=100)

    print("[INFO] Training model...")
    model, label_encoder, X_test, y_test, y_pred = train_model(X, y)

    print("[INFO] Showing predictions...")
    visualize_predictions(X_test, y_test, y_pred, label_encoder)
