
import os
from scipy.io import loadmat
import math
import numpy as np
import torch
from helpers import time_to_sec


def extract_sentence_features_tensor(i3d_mat_file_path, start_time, end_time, features_per_sec, sentence_no=''):
    i3d_mat_file = loadmat(i3d_mat_file_path)
    video_features_np = i3d_mat_file['preds']
    start_sec, end_sec = time_to_sec(start_time), time_to_sec(end_time)
    features_no_start = math.floor(start_sec*features_per_sec)
    features_no_end = math.ceil(end_sec*features_per_sec)
    sentence_features_np = video_features_np[features_no_start:features_no_end]
    sentence_features_tensor = torch.tensor(sentence_features_np, dtype=torch.float32)
    return sentence_features_tensor


# if __name__ == "__main__":
#     path = "/Users/chuntingjustinlo/Study/CardiffComputing/DissertationSignify/datasets/bobsl/features_5085344787448740525.mat"
#     tensor = extract_sentence_features_tensor(path, "00:01:00.800", "00:01:03.034", 6.25)
#     print(tensor)
#     print(tensor.dtype)