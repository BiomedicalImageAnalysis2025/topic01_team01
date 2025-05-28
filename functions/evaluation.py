import numpy as np
import matplotlib.pyplot as plt 
import functions.knn as knn
import functions.preprocessing as preprocessing

def model_evaluation(test_img, test_indices, train_img, train_labels, k):
    """
    test_img: Processed test images (PCA).
    test_indices: Original incices of the test images out of the complete dateset.
    train_img: Processed training images (PCA).
    train_labels: Labels of the training images to assign the test images to the k nearest neighbours.
    k: Number of nearest neighbours to consider for the KNN algorithm.
    """
    total_predictions = 0
    correct_predictions = 0 

    # We need the original indices (remember, the position number in the original dataset)
    # to compare the predictions with the original labels. -> enumerate assigns an index to each image.
    for i, test_img in enumerate(test_img):

        # Using the implemented kNN algorithm to predict the label of the test images by k neighbours.
        prediction = knn.kNN_predict(train_img, train_labels, test_img, k)

        total_predictions += 1

        # To get the person label 1 to 15, we need to compute the coresponding index.
        # By dividing (true division to get an int) each test index by 11, we get the the coresponding person ID.
        # The +1 is needed because the person IDs start at 1, not 0.
            # Note that this method only works because we know that each person has 11 images in the dataset and
            # we sorted them beforehand. 
        personID = (test_indices[i] // 11) + 1 

        if prediction == personID:

            correct_predictions += 1

        else:

            continue

    accuracy = correct_predictions / total_predictions

    print(f"Total predictions: {total_predictions}")
    print(f"Correct predictions: {correct_predictions}")
    print(100 * "---")
    print(f"Accuracy: {accuracy * 100:2f}%")

    return accuracy



