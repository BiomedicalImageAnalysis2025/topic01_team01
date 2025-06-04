pip ; install ; scikit-learn ; matplotlib ; opencv-python
import os
import cv2
import numpy as np
from sklearn.neighbors import train_test_split
from sklearn.preprocessing import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Path to Yale Face dataset
dataset_path = './dataset/' \

#Image settings
image_size = (100, 100) # Resize images to this size

X = [] 
y = []

# Load and preprocess images
for filename in os.listdir(dataset_path):
    if filename.endswith('.gif'):
       label =filename.split('subject01')[0]
       img_path = os.path.join(dataset_path, dataset)

       # Read image
       img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
       if img is None:
            continue
       
       # Resize and flatten
       img_resized = cv2.resize(img, image_size)
       img_flattened = img_resized.flatten()

       X.append(img_flattened)
       y.append(label)

# Convert to Numpy arrays
X = np.array(X)
y = np.array(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predict and evaluate
y_pred = knn.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("nClassification Report:\n", classification_report(y_test, y_pred))

# Optional: Visualize a few test images and predictions
for i in range(5):
    img = X_test[i].reshape(image_size)
    plt.imshow(img, cmap='gray')
    plt.title(f"True: {y_test[i]}, Pred: {y_test[i]}")
    plt.axis('off')
    plt.show()

        