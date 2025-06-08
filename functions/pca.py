# PCA (Principal Component Analysis) implementation
import os
import numpy as np
from PIL import Image

# Function to perform PCA on a dataset of images
def pca(train_data, n_components):
    """Compute the PCA transformation for the training set dat_matrix using SVD.

    Args:
        dat_matrix : ndarray of shape (n_samples, n_features)
            Training data.
        n_components : int
            Number of principal components to keep.

    Returns:
        U_reduced : ndarray of shape (n_samples, n_components)
            The left singular vectors (principal components) of the data matrix.
        S_reduced : ndarray of shape (n_components,)
            The singular values (square roots of eigenvalues) corresponding to the principal components.    
        V_reduced : ndarray of shape (n_components, n_features)
            The projection matrix containing the top principal component directions.
        eigenvalues : ndarray of shape (n_components,)
            The eigenvalues corresponding to the selected components (variance captured).
        train_reduced : ndarray of shape (n_samples, n_components)
            The training data projected onto the PCA space.
        variance_explained : ndarray of shape (n_components,)
            The ratio of variance explained by each principal component.
        n_components : int
            The number of principal components used in the PCA transformation.
    """
    # VT = Contains the right singular vectors (eigenvectors) -> quadratic matrix
    # we use full_matrices=False to get reduced matrices
    U, S, VT = np.linalg.svd(train_data, full_matrices=False)
    # selecting the first n_components
    U_reduced = U[:, :n_components]  # n_samples x n_components
    S_reduced = S[:n_components]     # n_components
    V_reduced  = VT[:n_components, :]

    # Be aware of the correct matrix mulitplication order.
    # Note that V_reduced returns as n x D matrix, where n is the number
    # of components and D the number of pixel but we need to multiply it with the train_data which
    # is N x D, where N is the number of images and D the number of pixels. 
    # -> For multiplication to work, we need to transpose V_reduced to D x n.
    
    # projecting the data onto the reduced space 
    train_reduced = train_data @ V_reduced.T

    # defining the eigenvalues 
    eigenvalues = S_reduced ** 2
    # calculating the variance explained by each component
    variance_explained = eigenvalues / np.sum(S ** 2)
    # returning reduced data
    print(f"Succesfully reduced Matrix from {train_data.shape} to {train_reduced.shape}\n")

    return U_reduced, S_reduced, V_reduced, train_reduced, eigenvalues, variance_explained, n_components

def pca_transform(test_data, V_reduced):
    """Transform the data matrix using the PCA projection matrix.

    Args:
        test_data: ndarray of shape (n_samples, n_features)
            The data to be transformed.
        V_reduced : ndarray of shape (n_components, n_features)
            The PCA projection matrix.

    Returns:
        test_reduced : ndarray of shape (n_samples, n_components)
            The transformed data in the PCA space.
    """       
    # Project the test data onto the PCA space using the projection matrix V_reduced
    test_reduced = test_data @ V_reduced.T
    print(f"Succesfully transformed Matrix from {test_data.shape} to {test_reduced.shape}\n")

    return test_reduced