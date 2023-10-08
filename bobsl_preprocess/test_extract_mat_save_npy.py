import numpy as np
from scipy.io import loadmat

# Load the .mat file
mat_file = loadmat('./5085344787448740525.mat')

data_tensor = mat_file['preds']

# Extract the (100, 1024) tensor
extracted_tensor = data_tensor[1:100]

# Save the extracted tensor to a new .npy file
np.save('5085344787448740525.npy', extracted_tensor)
