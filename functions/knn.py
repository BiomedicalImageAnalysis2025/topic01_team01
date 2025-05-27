import numpy as np
import matplotlib.pyplot as plt
from functions.preprocessing import train_centered, test_centered
from functions.pca import PCA, svd_for_pca

# As we will use the euclidean distance for the KNN
# Algorithm, we will define a function to calculate 
# the euclidean distance between two points.

def distance_euclidean(a,b):
    return np.square(np.sum(a - b)**2)


def kNN_predict(train_img, test_img, k, labels):
    # Set how many neighbours we want to consider.
    k = 10

    # Initilize an empty list to store the predictions.
    predicitions = []

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
        predicted_label = values[np.argmay(counts)]

        # Append the predicted label to the list.
        predicitions.append[predicted_label]

    return predicitions

