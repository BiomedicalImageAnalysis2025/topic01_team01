import os
import numpy as np
from PIL import Image


def preprocessing_split_ratio(train_ratio=0.8):
    # Path to the dataset folder. os.getcwd() gets the current working directory (in our case main.ipynb).
    folder_path = os.path.join(os.getcwd(), "datasets")

    # Create a dictionary to group images by individual.
    grouped_images = {}

    # Load images and group them by individual.
    for img_file in sorted(os.listdir(folder_path)):
        # Check if the file is a GIF image, if not, it will be skipped.
        if not img_file.endswith(".gif"):
            continue

        # Extract person’s identifier from the filename.
        subject_id = img_file.split(".")[0]
        
        # Open the image file and convert it to grayscale.
        img_path = os.path.join(folder_path, img_file)
        with Image.open(img_path) as img:
            img = img.convert("L")  # Convert to grayscale
            image_data = np.array(img, dtype=np.float32) / 255.0  # Normalize pixel values to [0, 1]
            flat_image = image_data.flatten()  # Flatten the 2D image array into a 1D vector
        
        # Group images by subject ID.
        if subject_id not in grouped_images:
            grouped_images[subject_id] = []
        grouped_images[subject_id].append(flat_image)  

    # Set a random seed for reproducibility (output is the same every time).
    np.random.seed(727)

    # Prepare separate lists for training and testing images.
    train_data = []
    test_data = []
    train_labels = []
    test_labels = []

    # For each individual, split images into training and testing sets based on the given ratio.
    for subject, images in grouped_images.items():
        images = np.array(images)  # Convert list of images to NumPy array
        np.random.shuffle(images)  # Shuffle images in-place

        # Calculate the split index based on the train_ratio
        split_index = int(len(images) * train_ratio)

        # Split into training and testing sets
        subject_train = images[:split_index]
        subject_test = images[split_index:]

        # Append images and corresponding labels
        train_data.extend(subject_train)
        train_labels.extend([subject] * len(subject_train))
        test_data.extend(subject_test)
        test_labels.extend([subject] * len(subject_test))

    # Output the total number of training and testing images
    print(f"Total training images: {len(train_data)}")
    print(f"Total testing images: {len(test_data)}")

    # Now Normalization & Standardization is performed

    # Convert lists to NumPy arrays.
    train_arr = np.array(train_data)  # Shape: (n_train, num_features)
    test_arr = np.array(test_data)   # Shape: (n_test, num_features)

    # Compute global mean and standard deviation from training data only.
    train_mean = np.mean(train_arr, axis=0)
    global_std = np.std(train_arr, axis=0)

    # To avoid division by zero, replace any zeros in the std vector.
    global_std[global_std == 0] = 1e-8

    # Center the training set.
    final_train = (train_arr - train_mean) / global_std

    # Apply the same transformation to the test set.
    final_test = (test_arr - train_mean) / global_std

    # Summary printout of the preprocessing steps
    print("\nAfter preprocessing:")
    print(f"Training data shape: {final_train.shape}")
    print(f"Testing data shape: {final_test.shape}")

    return final_train, train_labels, final_test, test_labels