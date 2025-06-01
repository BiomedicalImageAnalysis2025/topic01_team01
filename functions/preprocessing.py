# Here all the preprocessing steps are defined including splittin the dataset, to prepare the data for PCA.
# import necessary libraries
import os
import numpy as np
from PIL import Image

  
# Path to the dataset folder. os.getcwd() gets the current working directory.
# If The dataset folder is one level up from the current working directory,  use "../" before "datasets".
folder_path = os.path.join(os.getcwd(), "../datasets")

# Create a dictionary to group images by individual.
grouped_images = {}

# Load images and group them by individual.
for img_file in sorted(os.listdir(folder_path)):
    # Check if the file is a GIF image, if not, it will be skipped.
    if not img_file.endswith(".gif"):
        continue

    # Extract person’s identifier from the filename.
    # In our case, the identifier is the first part of the filename before the dot.
    # [0] splits the string at the dot and takes the first part.
    subject_id = img_file.split(".")[0]
    
    # Open the image file and convert it to grayscale.
    img_path = os.path.join(folder_path, img_file)
    with Image.open(img_path) as img:
        # The 'L' mode is for grayscale images.
        img = img.convert("L")
        # Convert the grayscale image to a NumPy array for numerical processing
        # Convert to a float32 NumPy array (this converts the integer data into float)
        image_data = np.array(img, dtype=np.float32)
        # Flatten the 2D image array into a 1D vector.
        flat_image = image_data.flatten()
    
    # Check if the subject_id is already in the dictionary.
    # If not, create a new list for that subject.
    # If it is, append the image data to the existing list.
    # This way, we group all images of the same subject together.
    if subject_id not in grouped_images:
        grouped_images[subject_id] = []
    grouped_images[subject_id].append(flat_image)  

# Prepare separate lists for training and testing images.
final_train = []
final_test = []

# For each individual, randomly assign 8 images to training and 3 to testing.
for subject, images in grouped_images.items():
    # Convert list of images to a numpy array for shuffling. 
    # This shuffling is along axis 0. (along rows, in our case images)
    images = np.array(images)
    
    # Shuffle images in-place with NumPy's shuffle.
    np.random.shuffle(images)
    
    # Create an array of labels: 8 'train' and 3 'test'.
    labels = np.array(['train'] * 8 + ['test'] * 3)
    
    # Shuffle the labels, so that the assignment order is random.
    np.random.shuffle(labels)
    
    # Now pair each image with a label.
    # zip combines the images and labels into pairs.
    for image, label in zip(images, labels):
        # distribute images to training and testing sets based on the label which is randomly assigned.
        if label == 'train':
            final_train.append(image)
        else:
            final_test.append(image)

# output you see in main.ipynb
print(f"Total training images: {len(final_train)}")
print(f"Total testing images: {len(final_test)}")

# Now Normalization & Standardization is performed

# Convert lists to NumPy arrays.
train_data = np.array(final_train)  # Shape: (n_train, num_features)
test_data  = np.array(final_test)   # Shape: (n_test, num_features)

# Compute global mean and standard deviation from training data only.
global_mean = np.mean(train_data, axis=0)

# Standardize the training set.
train_centered = (train_data - global_mean) 

# Apply the same transformation to the test set.
test_centered = (test_data - global_mean) 

# Summary printout of the preprocessing steps
print(f"Training data shape: {train_centered.shape}")
print(f"Testing data shape: {test_centered.shape}")

# For verification, print the mean and std of the first training image.
print(f"First training image: Mean ≈ {np.mean(train_centered[0]):.4f}")

#function runs only if this file is executed directly in main.ipynb
