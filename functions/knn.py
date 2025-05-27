import numpy as np
import os
import matplotlib.pyplot as plt

# As we will use the euclidean distance for the KNN
# Algorithm, we will define a function to calculate 
# the euclidean distance between two points.

def distance_euclidean(a,b):
    return np.square(np.sum(a - b)**2)


