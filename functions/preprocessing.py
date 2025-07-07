# Here all the preprocessing steps are defined including splitting the dataset
import os
import numpy as np
from PIL import Image

def preprocessing(data_path, seed, train_ratio= 0.7, verbose=True):
    """Preprocesses image data from the specified directory and splits it into training and testing sets.

    Each GIF image in the given directory is processed as follows:
      - Only files with the .gif extension are considered.
      - The subject's identifier is extracted from the filename (the portion before the first dot).
      - The image is opened, converted to grayscale ('L' mode), normalized (dividing pixel values by 255),
        and then flattened into a 1D NumPy array.
    
    Images for each subject are grouped together, then shuffled and split into training and testing sets.
    For each subject, the first 8 images (after shuffling) are used for training and the next 3 for testing.
    Finally, both training and testing images are standardized using the training set's mean 
    and standard deviation.

    Args:
        data_path (str): The path to the directory containing the image files.
        seed(int): A seed for random number generation to ensure reproducibility.
        train_ratio(float): The proportion of images to use for training.
        verbose (bool): If True, prints the number of training and testing images.

    Returns:
        final_train (numpy.ndarray): 
            The training data, standardized, with shape (n_train, num_features).
        final_test (numpy.ndarray):
            The testing data, standardized using the training set's parameters.
        train_labels (list):
            A list of subject identifiers corresponding to each training image.
        test_labels (list):
            A list of subject identifiers corresponding to each testing image.
        test_arr (numpy.ndarray):
            The original (non-standardized) testing data as a NumPy array.
    """
    
    # Set a random seed for reproducibility.
    np.random.seed(seed)

    # Create a dictionary to group images by individual.
    grouped_images = {}

    # Load images and group them by individual.
    for img_file in os.listdir(data_path):
        # Check if the file is a GIF image, if not, it will be skipped.
        if not img_file.endswith(".gif"):
            continue

        # Extract person’s identifier from the filename.
        # In our case, the identifier is the first part of the filename before the dot.
        # [0] splits the string at the dot and takes the first part.
        subject_id = img_file.split(".")[0]
        
        # Open the image file and convert it to grayscale.
        img_path = os.path.join(data_path, img_file)
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

    # Prepare separate lists for training and testing images.
    train_data = []
    test_data = []
    # Prepare labels lists, which will later be used for the KNN classifier.
    train_labels = []
    test_labels = []

    # For each individual, randomly assign 8 images to training and 3 to testing.
    for subject, images in grouped_images.items():
        
        # Shuffle images in-place with NumPy's shuffle.
        # This shuffling is along axis 0, meaning it shuffles the rows (images) randomly within the array.
        np.random.shuffle(images)

        # Calculate the split index based on the train_ratio
        split_index = int(len(images) * train_ratio)
        
        # Split into training and testing images
        subject_train = images[:split_index]
        subject_test = images[split_index:]

        # Append images and corresponding labels
        train_data.extend(subject_train)
        # Multiplying [subject] by len(subject_train) creates a list where the same subject ID repeats n times 
        # (where n is the number of training images for that subject).
        # This is used for the KNN classifier later.
        train_labels.extend([subject] * len(subject_train)) 
        #.extend is used to add elements of the list subject_train to train_data individually.
        # the output is one list, instead of a list of lists.(this would have been the case if we used .append)
        test_data.extend(subject_test)
        test_labels.extend([subject] * len(subject_test))

    if verbose:
        print(f"Total training images: {len(train_data)}")
        print(f"Total testing images: {len(test_data)}")


    #  Now Normalization & Standardization is performed

    # Convert lists to NumPy arrays.
    train_arr = np.array(train_data)  # Shape: (n_train, num_features)
    test_arr = np.array(test_data)   # Shape: (n_test, num_features)

    # Compute global mean and standard deviation from training data only.
    # axis = 0 means we compute the mean and std across all training images for each feature (pixel).
    train_mean = np.mean(train_arr, axis=0)
    global_std  = np.std(train_arr, axis=0)

    # To avoid division by zero, replace any zeros in the std vector.
    # np.where checks where global_std is zero and replaces it with a small value (1e-8).
    global_std = np.where(global_std == 0, 1e-8, global_std)

    # Center the training set.
    final_train = (train_arr - train_mean) / global_std

    # Apply the same transformation to the test set.
    final_test = (test_arr - train_mean) / global_std

    return final_train, train_labels, final_test, test_labels, test_arr