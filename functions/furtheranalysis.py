import scipy.io

def datasetB_loading():
    # Load the .mat file
    data = scipy.io.loadmat("C:datasetB\\YaleB_32x32.mat")

    # Check available keys
    print(data.keys())  # Lists variable names inside the .mat file
    # Extract one image (assuming they are grayscale)
    image = data["fea"]
    #image = data[fea[0]].reshape((32, 32))  # Adjust (height, width) based on actual dimensions

    return image, data
