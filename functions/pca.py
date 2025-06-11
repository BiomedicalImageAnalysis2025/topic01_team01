# PCA (Principal Component Analysis) implementation
import os
import numpy as np
from PIL import Image

# Function to perform PCA on a dataset of images
def svd_pca(input_matrix, n_components):
    """Compute the PCA transformation for the training set dat_matrix using SVD.

    Args:
        input_matrix : ndarray of shape (n_samples, n_features)
            Training data.
        n_components : int
            Number of principal components to keep.

    Returns:
        projection_matrix : ndarray of shape (n_components, n_features)
            The projection matrix containing the top principal component directions.
        train_reduced : ndarray of shape (n_samples, n_components)
            The training data projected onto the PCA space.
        explained_variance_ratio : ndarray of shape (n_components,)
            The ratio of variance explained by each principal component.
    """
    # VT = Contains the right singular vectors (eigenvectors) -> rectangular matrix (A^TA)^T	
    # U = Contains the left singular vectors (eigenvectors) -> rectangular matrix (AA^T)
    # we use full_matrices=False to get reduced matrices
    U, S, VT = np.linalg.svd(input_matrix, full_matrices=False)

    # S contains the singular values, which are the square roots of the eigenvalues and lie on the diagonal of matrix sigma
    # We take the first n_components singular values and corresponding vectors (projection matrix)
    singular_values = S[:n_components]
    projection_matrix  = VT[:n_components, :]

    # Be aware of the correct matrix mulitplication order.
    # Note that projection_matrix returns as n x D matrix, where n is the number
    # of components and D the number of pixel but we need to multiply it with the input_data which
    # is N x D, where N is the number of images and D the number of pixels. 
    # -> For multiplication to work, we need to transpose projection_matrix to D x n.
    
    # projecting the data onto the reduced space 
    train_reduced = input_matrix @ projection_matrix.T

    # defining the eigenvalues 
    eigenvalues = singular_values ** 2
    # calculating the variance explained by each component
    # we need to divide the eigenvalues by the total sum of eigenvalues to get the explained variance ratio
    # we convert the absolute eigenvalues to relative values
    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
    # returning reduced data
    print(f"Succesfully reduced Matrix from {input_matrix.shape} to {train_reduced.shape}\n")

    return projection_matrix, train_reduced, explained_variance_ratio

def pca_transform(test_data, projection_matrix):
    """Transform the data matrix using the PCA projection matrix.

    Args:
        test_data: ndarray of shape (n_samples, n_features)
            The data to be transformed.
        projection_matrix : ndarray of shape (n_components, n_features)
            The PCA projection matrix.

    Returns:
        test_reduced : ndarray of shape (n_samples, n_components)
            The transformed data in the PCA space.
    """       
    # Project the test data onto the PCA space using the projection matrix V_reduced
    test_reduced = test_data @ projection_matrix.T
    print(f"Succesfully transformed Matrix from {test_data.shape} to {test_reduced.shape}")

    return test_reduced