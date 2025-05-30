import os
import numpy as np
import matplotlib.pyplot as plt
import functions.preprocessing 


# Choosing the smallest number of k possible to capture the desired variance.
k = 100

def svd_for_pca(X, k):
    # X is the input data matrix (training or test set -> train_centered or test_centered).
    """
    Performs Singular Value Decomposition (SVD) on the input data matrix X
    k: Number of pcs to keep.
    """
    # No need for normalize the matrix as the SVD will automatically
    # calculate the direction of the highest variance.
    # U = Contains the left singular vectors (eigenfaces) -> quadratic matrix 
    # S = 1D Array of the singular values (variance) 
    # VT = Contains the right singular vectors (eigenvectors) -> quadratic matrix
    U, S, VT = np.linalg.svd(X, full_matrices = False)
    # Lets store the top k rows in a new matrix for PCA.
    projection_matrix = VT[ :k, : ]

    return projection_matrix, S


def PCA(projection_matrix, X):
    """
    Matrix multiplication of the input data matrix X with the projection_matrix
    to project the data into the PCA space.
    """
    # Be aware of the correct matrix mulitplication order.
    # Note that W returns as k x D matrix, where k is the number
    # of components and D the number of pixel but we need 
    # to multiply it with the X matrix which
    # is N x D, where N is the number of images and D the number of pixels. 
    # -> For multiplication to work, we need to transpose W to D x k.
    pca_dataset = X @ projection_matrix.T

    return pca_dataset
