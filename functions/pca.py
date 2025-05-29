# PCA (Principal Component Analysis) implementation
import os
import numpy as np
from PIL import Image

# Function to perform PCA on a dataset of images
# X is the input data matrix, n_components is the number of principal components to keep
def  pca(dat_matrix, n_components):
    
    # No need for normalize the matrix as the SVD will automatically
    # calculate the direction of the highest variance.
    # U = Contains the left singular vectors (eigenfaces) -> quadratic matrix 
    # S = 1D Array of the singular values (variance) 
    # VT = Contains the right singular vectors (eigenvectors) -> quadratic matrix
    # performing SVD
    #we use full_matrices=False to get reduced matrices
    U, S, VT = np.linalg.svd(dat_matrix, full_matrices=False)
    #selecting the first n_components
    U_reduced = U[:, :n_components]
    S_reduced = S[:n_components]
    V_reduced  = VT[:n_components, :]

    # Be aware of the correct matrix mulitplication order.
    # Note that W returns as k x D matrix, where k is the number
    # of components and D the number of pixel but we need 
    # to multiply it with the X matrix which
    # is N x D, where N is the number of images and D the number of pixels. 
    # -> For multiplication to work, we need to transpose W to D x k.
    
    #projecting the data onto the reduced space 
    dat_matrix_reduced = dat_matrix @ V_reduced.T

    #defining the eigenvalues 
    eigenvalues = S_reduced ** 2
    #returning reduced data

    print(f"Original Data Shape: {dat_matrix.shape}")
    print(f"Reduced Data Shape: {dat_matrix_reduced.shape}\n")

    return U_reduced, S_reduced, V_reduced, dat_matrix_reduced, eigenvalues