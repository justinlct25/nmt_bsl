
import os
from process_videos import process_sign_videos_folder, process_sign_videos_folder_multi_process
from operations_tsv import tsv_file_create


def create_output_folders(output_folder="./dataset_preprocess_output", 
                          npy_folder_name="sentence_sign_i3d_features",
                          tsv_file_name="i3d.bslcp.tsv"):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    npy_folder = os.path.join(output_folder, npy_folder_name)
    if not os.path.exists(npy_folder):
        os.mkdir(npy_folder)
    tsv_file = os.path.join(output_folder, tsv_file_name)
    tsv_file_create(tsv_file)
    return npy_folder, tsv_file
    

def list_video_folders(dataset_folder, video_extensions):
    video_folders = []
    for root, dirs, files in os.walk(dataset_folder):
        if any(file.endswith(tuple(video_extensions)) for file in files):
            video_folders.append(root)
    return video_folders

def process_video_data_folders(videos_folders_list, model, npy_folder, tsv_file, multi_process=False):
    for videos_folder in videos_folders_list:
        folder_name = os.path.basename(videos_folder)
        if "filtered" in folder_name:
            continue
        if multi_process:
            process_sign_videos_folder_multi_process(videos_folder, model, npy_folder, tsv_file, folder_name, is_features_per_frame=True)
        else:
            process_sign_videos_folder(videos_folder, model, npy_folder, tsv_file, folder_name)

