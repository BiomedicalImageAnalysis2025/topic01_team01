import os
import numpy as np
from PIL import Image
# for K-Nearest Neighbors (KNN) algorithm, a distance function is determine nearest neighbors
#def distance(a, b):
   # """Calculate the Euclidean distance between two images.

    #Args:
     #   a (_type_): _description_
      #  b (_type_): _description_

    # Returns:
    #    _type_: _description_
    # """
    # return np.sqrt(np.sum((a - b) ** 2))

def knn_classifier(train_reduced, train_labels, test_reduced, k):
    """
    A one-function KNN classifier that predicts the label for each test instance.
    
    Args:
        train_data (numpy.ndarray): 2D array where each row is a training sample.
        train_labels (list or numpy.ndarray): Labels corresponding to each training sample.
        test_data (numpy.ndarray): 2D array where each row is a test sample.
        k (int): Number of nearest neighbors to consider.
    
    Returns:
        list: Predicted labels for each test sample.
    """
    predictions = []
    
    for test_point in test_reduced:
        # Compute Euclidean distances between the test point and all training samples.
        # axis=1 ensures that we compute the distance for each row (sample).
        # Using vectorized operations for efficiency.
        distances = np.sqrt(np.sum((train_reduced - test_point) ** 2, axis=1))
        
        # Get the indices of the k smallest distances.
        k_indices = np.argsort(distances)[:k]
        
        # Retrieve the labels for these k nearest neighbors.
        k_labels = [train_labels[i] for i in k_indices]
        
        # Majority vote: select the label with the highest count.
        prediction = max(set(k_labels), key=k_labels.count)
        predictions.append(prediction)
    
    return predictions

def knn_predict(train_imgs, labels, test_img, k):
    """_summary_

    Args:
        train_imgs (_type_): _description_
        labels (_type_): _description_
        test_img (_type_): _description_
        k (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Get the similarities of all training images to the test image.
    distances = [ distance(test_img, train_img) for train_img in train_imgs ]
    
    # Select the top k indices of the images with the smallest "distance"
    # to the test image.
    k_indices = np.argsort(distances)[:k]
    
    # Get the subject indices from the labels using the
    # indices of the most fitting images.
    k_labels = labels[k_indices]
    
    # Get all subjectIDs as well as the amount they occur.
    values, counts = np.unique(k_labels, return_counts=True)
    
    # Return the subjectID with the greatest amount.
    return values[np.argmax(counts)]
