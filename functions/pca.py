# Here all the preprocessing steps are defined including splittin the dataset, to prepare the data for PCA.
# import necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import functions.preprocessing 


# Choosing the smallest number of k possible to capture the desired variance.
k = 100

def svd_for_pca(X, k):
    # X is the input data matrix (training or test set -> train_centered or test_centered).

    # No need for normalize the matrix as the SVD will automatically
    # calculate the direction of the highest variance.
    # U = Contains the left singular vectors (eigenfaces) -> quadratic matrix 
    # S = 1D Array of the singular values (variance) 
    # VT = Contains the right singular vectors (eigenvectors) -> quadratic matrix
    U, S, VT = np.linalg.svd(X, full_matrices = False)
    # Lets store the top k rows in a new matrix for PCA.
    W_dataset = VT[ :k, : ]

    return W_dataset, S


def PCA(W_dataset, X):

    # Be aware of the correct matrix mulitplication order.
    # Note that W returns as k x D matrix, where k is the number
    # of components and D the number of pixel but we need 
    # to multiply it with the X matrix which
    # is N x D, where N is the number of images and D the number of pixels. 
    # -> For multiplication to work, we need to transpose W to D x k.
    pca_dataset = X @ W_dataset.T

    return pca_dataset
