import os
import numpy as np
from PIL import Image
# for K-Nearest Neighbors (KNN) algorithm, a distance function is determine nearest neighbors
def knn_classifier(train_data, train_labels, test_data, k):
    """
    k-Nearest Neighbors (KNN) classifier implementation.

    Args:
      train_data (np.ndarray): 2D array of training data points, each row is a data point.
      train_labels (list or np.ndarray): List of labels for the training data.
      test_data (np.ndarray): 2D array of test data points.
      k (int): Number of nearest neighbors to consider.

    Returns:
      list: Predicted labels for each test data point.
    """
    predictions = []
    
    # Loop over each test sample
    for test_point in test_data:
        # Compute Euclidean distances between the test point and all training samples.
        # axis=1 ensures that we compute the distance for each row (sample).
        # The subtraction is vectorized over the training data for efficiency.
        distances = np.sqrt(np.sum((train_data - test_point) ** 2, axis=1))
        
        # Retrieve the indices of the k smallest distances.
        k_indices = np.argsort(distances)[:k]
        
        # Retrieve the labels for these k nearest neighbors.
        k_labels = [train_labels[i] for i in k_indices]
        
        # Make a prediction by selecting the most common label among the neighbors.
        # In case of a tie, max(..., key=k_labels.count) returns one of them.
        predicted_label = max(set(k_labels), key=k_labels.count)
        
        predictions.append(predicted_label)
    
    return predictions
