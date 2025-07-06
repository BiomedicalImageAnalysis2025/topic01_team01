# PCA (Principal Component Analysis) implementation
import os
import numpy as np
from PIL import Image

# Function to perform PCA on a dataset of images
def svd_pca(input_matrix, n_components, verbose=True):
    """
    Perform Principal Component Analysis (PCA) using Singular Value Decomposition (SVD).

    This function computes the PCA transformation of the input data matrix by applying SVD.
    It returns the projection matrix, the reduced representation of the input data, and the
    explained variance ratio of the selected principal components.

    Args:
        input_matrix (np.ndarray): 
            A 2D array of shape (n_samples, n_features) representing the input data.
        n_components (int): 
            The number of principal components to retain.
        verbose (bool, optional): 
            If True, prints the shape of the reduced matrix. Defaults to True.

    Returns:
        projection_matrix (np.ndarray): 
            A 2D array of shape (n_components, n_features) representing the PCA projection matrix.
        train_reduced (np.ndarray): 
            A 2D array of shape (n_samples, n_components) representing the input data projected into PCA space.
        explained_variance_ratio (np.ndarray): 
            A 1D array of shape (n_components,) containing the proportion of variance explained by each component.
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
    if verbose:
        print(f"\nSuccesfully reduced matrix from {input_matrix.shape} to {train_reduced.shape}\n")
        
    return projection_matrix, train_reduced, explained_variance_ratio

def pca_transform(test_data, projection_matrix, verbose=True):
    """
    Project new data into PCA space using a precomputed projection matrix.

    This function applies the PCA transformation to new data using the projection matrix
    obtained from a previous PCA fit.

    Args:
        test_data (np.ndarray): 
            A 2D array of shape (n_samples, n_features) representing the data to transform.
        projection_matrix (np.ndarray): 
            A 2D array of shape (n_components, n_features) representing the PCA projection matrix.
        verbose (bool, optional): 
            If True, prints the shape of the transformed matrix. Defaults to True.

    Returns:
        test_reduced (np.ndarray): 
            A 2D array of shape (n_samples, n_components) representing the transformed data in PCA space.
    """       
    test_reduced = test_data @ projection_matrix.T

    if verbose:
        print(f"Succesfully transformed matrix from {test_data.shape} to {test_reduced.shape}")

    return test_reduced