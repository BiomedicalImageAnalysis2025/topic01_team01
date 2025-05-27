# Here all the preprocessing steps are defined including splittin the dataset, to prepare the data for PCA.
# import necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from functions.preprocessing import train_centered


# Choosing the smallest number of k possible to capture the desired variance.
k = 100

def svd_for_pca(train_centered_svd, k):
    # No need for normalize the matrix as the SVD will automatically
    # calculate the direction of the highest variance.
    # U = N x k
    # S = 1D Array of the singular values (variance)
    # VT = k x D (eigenfaces)
    U, S, VT = np.linalg.svd(train_centered_svd, full_matrices = False)


    # Lets store the top k rows in a new matrix for PCA.
    W_train = VT[ :k, : ]

    return W_train

def PCA(W_train, train_centered_svd):

    # Be aware of the correct matrix mulitplication order.
    # Note that W returns as k x D matrix, where k is the number of components and D the number of pixels
    # but we need to multiply it with the train_centered_svd matrix which is N x D, where N is the number 
    # of images and D the number of pixels. 
    # -> For multiplication to work, we need to transpose W to D x k.
    pca_train = train_centered_svd @ W_train.T

    return pca_train
