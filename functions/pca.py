import os
import numpy as np
import matplotlib.pyplot as plt

# Choosing the smallest number of n_components possible to capture the desired variance.
n_components = 15

def svd_for_pca(X, n_components):
    # X is the input data matrix (training or test set -> train_centered or test_centered).
    """
    Performs Singular Value Decomposition (SVD) on the input data matrix X
    n_components: Number of pcs to keep.
    """
    # No need for normalize the matrix as the SVD will automatically
    # calculate the direction of the highest variance.
    # U = Contains the left singular vectors (eigenfaces) -> quadratic matrix 
    # S = 1D Array of the singular values (variance) 
    # VT = Contains the right singular vectors (eigenvectors) -> quadratic matrix
    U, S, VT = np.linalg.svd(X, full_matrices = False)
    # Lets store the top n_components rows in a new matrix for PCA.
    projection_matrix = VT[ :n_components, : ]
    singular_values = S[ :n_components]    


    eigenvalues = singular_values ** 2
    variance_explained = eigenvalues / np.sum(S ** 2)
    # Show the variance captured by each component.
    print(f"Explained variance ratio by first 15 components:\n {variance_explained[:15]}\n")
    # If the summed variance of n_components explains more than 85% of the total variance, we can consider this a good choice.
    print(f"Variance captured by the first {n_components} components: {np.sum(S[:n_components]) / np.sum(S) * 100:.2f}%")

    return projection_matrix, singular_values, variance_explained, eigenvalues


def PCA(projection_matrix, X):
    """
    Matrix multiplication of the input data matrix X with the projection_matrix
    to project the data into the PCA space.
    X: Input data matrix (training or test set -> train_centered or test_centered).
    """
    # Be aware of the correct matrix mulitplication order.
    # Note that W returns as  x D matrix, where n_components is the number
    # of components and D the number of pixel but we need 
    # to multiply it with the X matrix which
    # is N x D, where N is the number of images and D the number of pixels. 
    # -> For multiplication to work, we need to transpose W to D x n_components.
    pca_dataset = X @ projection_matrix.T

    # The shape should be now 120 (for training) and 45 (for testing) x n_components.
    print(f"Succesfully transformed Matrix from {X.shape} to {pca_dataset.shape}\n")

    return pca_dataset
