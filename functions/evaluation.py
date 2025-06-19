# Here all the preprocessing steps are defined including splitting the dataset
import os
import numpy as np
from PIL import Image
import pandas as pd

def metadata(data_path):

    exp_cond = []

    for img_file in os.listdir(data_path):
        # Check if the file is a GIF image, if not, it will be skipped.
        if not img_file.endswith(".gif"):
            continue
        # Extract person’s identifier from the filename.
        # I our case, the identifier is the first part of the filename before the dot.
        # [0] splits the string at the dot and takes the first part.
        exp_cond_single = img_file.split(".")[1]
        exp_cond.append(exp_cond_single)

        # Separate lists
    condition_meta = [f for f in exp_cond if "light" in f.lower()]
    rest_meta = [f for f in exp_cond if "light" not in f.lower()]

    # Create DataFrame
    metadata_A = pd.DataFrame({
        "light_related": pd.Series(condition_meta),
        "others": pd.Series(rest_meta)
    })

    return metadata_A

def metadata_sub(data_path):

    subject_id = []

    for img_file in os.listdir(data_path):
        # Check if the file is a GIF image, if not, it will be skipped.
        if not img_file.endswith(".gif"):
            continue
        # Extract person’s identifier from the filename.
        # I our case, the identifier is the first part of the filename before the dot.
        # [0] splits the string at the dot and takes the first part.
        subject_id_sing = img_file.split(".")[0]
        subject_id.append(subject_id_sing)
    
    # Create DataFrame
    metadata_sub = pd.DataFrame({
        "Subject ID": subject_id
    })

    return metadata_sub