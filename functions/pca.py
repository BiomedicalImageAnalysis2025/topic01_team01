# PCA (Principal Component Analysis) implementation
import os
import numpy as np
from PIL import Image

def  pca(X, n_components):
    # X is the input data matrix (training or test set -> train_centered or test_centered).
    # No need for normalize the matrix as the SVD will automatically
    # calculate the direction of the highest variance.
    # U = Contains the left singular vectors (eigenfaces) -> quadratic matrix 
    # S = 1D Array of the singular values (variance) 
    # VT = Contains the right singular vectors (eigenvectors) -> quadratic matrix
    #performing SVD
    #we use full_matrices=False to get reduced matrices
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    #selecting the first n_components
    eigenfaces = U[:, :n_components]
    S_reduced = S[:n_components]
    V_reduced  = Vt[:n_components, :]

    
    #projecting the data onto the reduced space 
    X_reduced = X @ V_reduced.T

    #defining the eigenvalues 
    eigenvalues = S_reduced ** 2
    #returning reduced data
    return eigenfaces, S_reduced, V_reduced, X_reduced, eigenvalues