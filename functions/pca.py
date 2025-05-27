# functions to be used in the main.py file
import os
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandernScaler
from sklearn.metrics import classificaation_report

# Set image size (must match all images)
IMG_SIZE = (100, 100)

#Path to your dataset folder
DATASET_DIR = 'dataset/'

X = []  # image data
y = []  # labels

# Load images and extract labels
for filename in os.listdir(DATASET_DIR)
    if filename.endswith('.gif') 
        # Extract subject number from filename (e.g., 'subject01' -> 1)
        subject_id = int(filename.split('subject')[1].split('_')[0])

        # Load image in grayscale and resize
        img_path = os.path.join(DATASET_DIR, datasets)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, IMG_SIZE)

        # Flatten image to 1D vector
        X.append(img.flatten())
        y.append(subject_id)

X = np.array(X)
y = np.array(y)

# Standardize the data
scaler = StandardScaler
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=100, whiten=True)
X_pca =.fit_transform(X_scaled)

# Split data into train/test
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Train classifier (e.g., linear SVM)
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

#Predict and evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

#Show some Eigenfaces
import matplotlib.pyplot as plt

eigenfaces = pca.components_.reshape((100, IMG_SIZE[0], IMG_SIZE[1]))
for i in range(10):
    plt.imshow(eigenfaces[i], cmap='gray')
    plt.title(f"Eigenface {i+1}")
    plt.axis('off')
    plt.show()