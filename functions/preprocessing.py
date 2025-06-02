# Here all the preprocessing steps are defined including splittin the dataset, to prepare the data for PCA.
# import necessary libraries
import os
import numpy as np
from PIL import Image

# Path to the dataset folder. os.getcwd() gets the current working directory (in our case main.ipynb).
# If The dataset folder is one level up from the current working directory,  use "../" before "datasets".
folder_path = os.path.join(os.getcwd(), "datasets")

# Create a dictionary to group images by individual.
grouped_images = {}

# Load images and group them by individual.
for img_file in sorted(os.listdir(folder_path)):
    # Check if the file is a GIF image, if not, it will be skipped.
    if not img_file.endswith(".gif"):
        continue

    # Extract person’s identifier from the filename.
    # I our case, the identifier is the first part of the filename before the dot.
    # [0] splits the string at the dot and takes the first part.
    subject_id = img_file.split(".")[0]
    
    # Open the image file and convert it to grayscale.
    img_path = os.path.join(folder_path, img_file)
    with Image.open(img_path) as img:
        # The 'L' mode is for grayscale images.
        img = img.convert("L")
        # Convert the grayscale image to a NumPy array for numerical processing
        # Convert to a float32 NumPy array (this converts the integer data into float)
        # then normalize by dividing by 255 so that pixel values are in [0,1]
        image_data = np.array(img, dtype=np.float32) / 255.0
        # Flatten the 2D image array into a 1D vector.
        flat_image = image_data.flatten()
    
    # Check if the subject_id is already in the dictionary.
    # If not, create a new list for that subject.
    # If it is, append the image data to the existing list.
    # This way, we group all images of the same subject together.
    if subject_id not in grouped_images:
        grouped_images[subject_id] = []
    grouped_images[subject_id].append(flat_image)  

# Set a random seed for reproducibility (output is the same every time).
np.random.seed(165)

# Prepare separate lists for training and testing images.
train_data = []
test_data = []
#labels are later used for the KNN classifier.
train_labels = []
test_labels = []

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
            train_data.append(image)
            train_labels.append(subject)
        else:
            test_data.append(image)
            test_labels.append(subject)

# output you see in main.ipynb
print(f"Total training images: {len(train_data)}")
print(f"Total testing images: {len(test_data)}")

# Now Normalization & Standardization is performed

# Convert lists to NumPy arrays.
train_arr = np.array(train_data)  # Shape: (n_train, num_features)
test_arr = np.array(test_data)   # Shape: (n_test, num_features)

# Compute global mean and standard deviation from training data only.
train_mean = np.mean(train_arr, axis=0)
#global_std  = np.std(train_arr, axis=0)

# To avoid division by zero, replace any zeros in the std vector.
#global_std[global_std == 0] = 1e-8

# Center the training set.
final_train = (train_arr - train_mean) #/ global_std is used for standardization, but here we only center the data.

# Apply the same transformation to the test set.
final_test = (test_arr - train_mean) #/ global_std

# Summary printout of the preprocessing steps
#\n is used to have a space between the output of the two print statements.
print("\nAfter preprocessing:")
print(f"Training data shape: {final_train.shape}")
print(f"Testing data shape: {final_test.shape}")

# For verification, print the mean and std of the first training image.
# , Std ≈ {np.std(final_train[0]):.4f}
print(f"First training image: Mean ≈ {np.mean(final_train[0]):.4f}")