import os
import numpy as np
import matplotlib.pyplot as plt

# As we will use the euclidean distance for the KNN
# Algorithm, we will define a function to calculate 
# the euclidean distance between two points.

def distance_euclidean(a,b):
    """
    Returns the euclidean distance between two points a and b.
    """
    return np.sqrt(np.sum((a - b)**2))


# Set how many neighbours we want to consider.
# k = 1, 2, 3, 4, 5, etc.


def kNN_predict(train_img, test_img, labels, k):
    """
    Calculates the distance between each test image and ALL training images.
    train_img: List of processed training images.
    test_img: List of processed test images.
    labels: Labels of the training images.
    k: Number of nearest neighbours to consider.
    """
    # Initilize an empty list to store the predictions.
    predicitions = []

    # Loop through each test image.
    for test_img in test_img:

        # Calculate the distance between the test image an all training images.
        distance = [distance_euclidean(test_img, train_img) for train_img in train_img]

        # Sort the distances and get the smallest k indices -> nearest neighbours.
        k_indices = np.argsort(distance)[ :k]

        # Get the labels of these neighbours.
        k_labels = labels[k_indices]

        # Count how often each label appears.
        counts, values = np.unique(k_labels, return_counts = True)

        # Get the label with the highest count.
        predicted_label = values[np.argmax(counts)]

        # Append the predicted label to the list.
        predicitions.append(predicted_label)

    return predicitions
