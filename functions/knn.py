# Here, the implementation of the k-Nearest Neighbors (KNN) algorithm can be found.
import os
import numpy as np
from PIL import Image
from sklearn.neighbors import KDTree

def knn_classifier(train_reduced, train_labels, test_reduced, test_labels, k, verbose=True):
    """
    k-Nearest Neighbors (KNN) classifier implementation.

    Args:
      train_reduced (np.ndarray):
        2D array of training data points, each row is a data point.
      train_labels (list or np.ndarray):
        List of labels for the training data.
      test_reduced (np.ndarray):
        2D array of test data points.
      test_labels(list or np.ndarray):
        List of labels for the test data.
      k (int):
        Number of nearest neighbors to consider.
      verbose (boolean):
        Regulation of function print output. (Standard verbose = True)

    Returns:
      list: Predicted labels for each test data point.
    """
    # Fit a KDTree to the training data for efficient nearest neighbor search.
    # KDTree is a data structure that allows for efficient nearest neighbor searches.
    tree = KDTree(train_reduced,leaf_size=10)

    # Query the KDTree for the k nearest neighbors of each test point.
    distances, indices = tree.query(test_reduced, k=k, workers = -1)

    predictions = []
    # Loop over each test sample
    for idx_list in indices:
        
        #idx_list is a list of indices of the k nearest neighbors in the training set
        # Retrieve the labels for these k nearest neighbors.
        k_labels = [train_labels[i] for i in idx_list]
        
        predicted_label = max(set(k_labels), key=k_labels.count)
        predictions.append(predicted_label)

    # ... == ... creates a Boolean NumPy array where each element is True if the predicted label matches the true label, and False otherwise
    # and compares the predicted labels with the actual test labels
    # lastly mean() calculates the proportion of correct predictions
    accuracy = np.mean(np.array(predictions) == np.array(test_labels))
    # 100:.2 formats the accuracy as a percentage with two decimal places
    if verbose:
      print(f"k-NN Classification Accuracy: {accuracy * 100:.2f}%")
    return predictions, accuracy

  # IMPORTNAT NOTE:

  # Before using KDtrees, we computed the Euclidean distance manually
  # As we need faster perfromance, especially to create complex plots, we use KDtrees

  # Compute Euclidean distances between the test point and all training samples.
  # axis=1 ensures that we compute the distance for each row (sample) through the columns (features).
  # The subtraction is vectorized over the training data for efficiency.
  # distances = np.sqrt(np.sum((train_reduced - test_point) ** 2, axis=1))