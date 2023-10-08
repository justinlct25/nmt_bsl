import json
import os
import numpy as np
import scipy.io as sio

def sum_total_row_number_of_tensors(folder_path):
    total_rows = 0

    # Get a list of all files in the folder
    files = os.listdir(folder_path)

    for filename in files:
        if filename.endswith(".mat"):
            file_path = os.path.join(folder_path, filename)

            # Load the .mat file
            mat_data = sio.loadmat(file_path)

            # Assuming each file contains a tensor with the same key 'data' (you can change it based on your data structure)
            tensor = mat_data['preds']

            # Get the number of rows in the tensor and add it to the total
            total_rows += tensor.shape[0]

    return total_rows

# Replace 'folder_path' with the path to your folder containing .mat files
with open('config.json') as config_file:
        config_data = json.load(config_file)
I3D_FEATURES_PATH = config_data["I3D_FEATURES_PATH"]
total_rows_sum = sum_total_row_number_of_tensors(I3D_FEATURES_PATH)
print("Total row number of tensors in the folder:", total_rows_sum)
print("Total seconds of sentences: ", total_rows_sum/6.25)
