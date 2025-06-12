# Here, the implementation of the k-Nearest Neighbors (KNN) algorithm can be found.
import os
import numpy as np
from PIL import Image

def knn_classifier(train_reduced, train_labels, test_reduced,test_labels, k, verbose=True):
    """
    k-Nearest Neighbors (KNN) classifier implementation.

    Args:
      train_data (np.ndarray):
        2D array of training data points, each row is a data point.
      labels (list or np.ndarray):
        List of labels for the training data.
      test_data (np.ndarray):
        2D array of test data points.
      k (int):
        Number of nearest neighbors to consider.
      verbose (boolean):
        Regulation of function print output. (Standard verbose = True)

    Returns:
      list: Predicted labels for each test data point.
    """
    predictions = []
    # Loop over each test sample
    for test_point in test_reduced:
        # Compute Euclidean distances between the test point and all training samples.
        # axis=1 ensures that we compute the distance for each row (sample) through the columns (features).
        # The subtraction is vectorized over the training data for efficiency.
        distances = np.sqrt(np.sum((train_reduced - test_point) ** 2, axis=1))
        
        # Retrieve the indices of the k smallest distances.
        k_indices = np.argsort(distances)[:k]

        # Retrieve the labels for these k nearest neighbors.
        k_labels = [train_labels[i] for i in k_indices]
        
         # Get the labels of the k nearest neighbors.
        #counts, values = np.unique(k_labels, return_counts = True)

        # Get the label with the highest count.
        #predicted_label = values[np.argmax(counts)]
        predicted_label = max(set(k_labels), key=k_labels.count)
        
        predictions.append(predicted_label)

    # ... == ... creates a Boolean NumPy array where each element is True if the predicted label matches the true label, and False otherwise
    # and compares the predicted labels with the actual test labels
    # lastly mean() calculates the proportion of correct predictions
    accuracy = np.mean(np.array(predictions) == np.array(test_labels))
    # 100:.2 formats the accuracy as a percentage with two decimal places
    if verbose:
      print(f"k-NN Classification Accuracy: {accuracy * 100:.2f}%")
    return predictions
