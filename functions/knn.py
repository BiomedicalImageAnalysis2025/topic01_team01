# Here, the implementation of the k-Nearest Neighbors (KNN) algorithm can be found.
import os
import numpy as np
from PIL import Image
from sklearn.neighbors import KDTree

def knn_classifier(train_reduced, train_labels, test_reduced, test_labels, k, verbose=True):
  """
  Perform classification using the k-Nearest Neighbors (k-NN) algorithm.

  This function uses a KDTree for efficient nearest neighbor search to classify
  test samples based on the majority label of their k nearest neighbors in the
  training set.

  Args:
      train_reduced (np.ndarray): 
          A 2D array of shape (n_train_samples, n_features) representing the reduced training data.
      train_labels (list or np.ndarray): 
          Labels corresponding to the training data.
      test_reduced (np.ndarray): 
          A 2D array of shape (n_test_samples, n_features) representing the reduced test data.
      test_labels (list or np.ndarray): 
          Ground truth labels for the test data.
      k (int): 
          The number of nearest neighbors to consider for classification.
      verbose (bool, optional): 
          If True, prints the classification accuracy. Defaults to True.

  Returns:
      predictions (list): 
          Predicted labels for each test data point.
      accuracy (float): 
          Classification accuracy as a float between 0 and 1.
  """
  
  predictions = []

  for test_image in test_reduced:

    # Compute Euclidean distances between the test point and all training samples.
    # axis=1 ensures that we compute the distance for each row (sample) through the columns (features).
    # The subtraction is vectorized over the training data for efficiency.
    distances = np.sqrt(np.sum((train_reduced - test_image) ** 2, axis=1))

    # Get the indices of the k nearest neighbors based on the computed distances.
    k_indices = np.argsort(distances)[:k]

    k_labels = [train_labels[i] for i in k_indices]
 
    # if there is a tie, the max function will return the first encountered label
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

  # We also tried using KDtrees for faster perfromance, especially to create complex plots
  # As the code runs also very fast without KDTree, we decided to not use it in the end.
  # Still, the code we implemented before is shown below for reference.

  # Fit a KDTree to the training data for efficient nearest neighbor search.
  # KDTree is a data structure that allows for efficient nearest neighbor searches.
  # tree = KDTree(train_reduced,leaf_size=10)

  # Query the KDTree for the k nearest neighbors of each test point.
  # distances, indices = tree.query(test_reduced, k=k)

  # Loop over each test sample
  # for idx_list in indices:
       
      #idx_list is a list of indices of the k nearest neighbors in the training set
      # Retrieve the labels for these k nearest neighbors.
      #k_labels = [train_labels[i] for i in idx_list]