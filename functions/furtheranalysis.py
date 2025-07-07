import scipy.io
import numpy as np

def datasetB_loading():

    # Load the .mat file
    dataset_B = scipy.io.loadmat("dataB/YaleB_32x32.mat")

    # store images in images and their labels in true_label_B
    images = dataset_B["fea"]
    true_labels_B = dataset_B["gnd"]

    images_B = []

    # reshape all images and store them in images_B 
    for i in range(0,2414): 
        image_reshaped = images[i].reshape(32, 32)
        images_B.append(image_reshaped)
    
    images_B = np.array(images_B)

    return images_B, true_labels_B

def preprocessing_dataset_B(images, labels, seed, train_ratio, verbose=True):

    # Set a random seed for reproducibility (output is the same every time).
    np.random.seed(seed)
    images_flat = np.array([img.flatten() for img in images])

    # Create an array of indices and shuffle it
    indices = np.arange(images_flat.shape[0])
    np.random.shuffle(indices)

    # Rearrange images and labels using the same random permutation
    images_shuffled = images_flat[indices]
    labels_shuffled = labels[indices]

    # Calculate the split index based on the train_ratio
    split_index = int(images_flat.shape[0]*train_ratio)

    # Create training and test splits
    train_data = images_shuffled[:split_index]
    test_data = images_shuffled[split_index:]
    train_labels = labels_shuffled[:split_index]
    test_labels = labels_shuffled[split_index:]

    if verbose:
        print(f"Total training images: {len(train_data)}")
        print(f"Total testing images: {len(test_data)}")

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
    final_train_B = (train_arr - train_mean) / global_std

    # Apply the same transformation to the test set.
    final_test_B = (test_arr - train_mean) / global_std

    # Converts np arrays to lists, where each label is converted to a scalar for KNN classifier. 
    train_labels_B = [train_label.item() if isinstance(train_label, np.ndarray) else train_label for train_label in train_labels]
    test_labels_B = [test_label.item() if isinstance(test_label, np.ndarray) else test_label for test_label in test_labels]

    return final_train_B, train_labels_B, final_test_B, test_labels_B