import numpy as np

# Path to the .npy file
# file_path = './how2sign/_fZbAxSSbX4_0-5-rgb_front.npy'
# file_path = './how2sign/-fZc293MpJk_0-1-rgb_front.npy'
# file_path = './how2sign/-fZc293MpJk_2-1-rgb_front.npy'
# file_path = './how2sign/00dWJ4YRRSI/00dWJ4YRRSI_9-1-rgb_front.npy'
# file_path = './5085344787448740525.npy'
# file_path = "./dataset_preprocess_output/sentence_sign_i3d_features/N30i/N30i_1.npy"
file_path = "./dataset_preprocess_output/sentence_sign_i3d_features/N25i/N25i_1.npy"
# file_path = "./dataset_preprocess_output_interview_20230811/sentence_sign_i3d_features/N25i/N25i_1.npy"


# Read the .npy file
data = np.load(file_path)

# Access the contents of the .npy file
print(data.shape)
print(data)


# 22.378, 25.032

# frames: 139, 156
