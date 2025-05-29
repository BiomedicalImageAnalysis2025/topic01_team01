# PCA (Principal Component Analysis) implementation
import os
import numpy as np
from PIL import Image

def  pca(X, n_components):
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