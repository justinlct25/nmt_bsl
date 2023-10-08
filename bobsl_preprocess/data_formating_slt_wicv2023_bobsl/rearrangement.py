import json
import os
from datetime import datetime
import re
import csv
from scipy.io import loadmat
import math
import numpy as np


## according to the episodes with manually-aligned subtitle of BOBSL
## rearrange the i3d features of sign video frames to the arrangement of How2Sign for training the slt_wicv2023 SLT model
## create npy files of sentence and a tsv

def is_time_line(line):
    time_pattern = r"\d{2}:\d{2}:\d{2}\.\d{3}"
    return re.match(time_pattern, line)

def time_to_second(time_str):
    time_format = "%H:%M:%S.%f"
    time_obj = datetime.strptime(time_str, time_format)
    total_seconds = (time_obj.hour * 3600) + (time_obj.minute * 60) + time_obj.second + (time_obj.microsecond / 1e6)
    return total_seconds

def extract_start_end_sec(time_line): # eg. "00:00:25.082 --> 00:00:27.482"
    start_time, end_time = time_line.strip().split(' --> ')
    return time_to_second(start_time), time_to_second(end_time)

def extract_sentence_features_matrix(i3d_mat_file_path, start_sec, end_sec, features_per_sec, sentence_no, numpy_folder_path):
    i3d_mat_file = loadmat(i3d_mat_file_path)
    features_tensor = i3d_mat_file['preds']
    features_no_start = math.floor(start_sec*features_per_sec)
    features_no_end = math.ceil(end_sec*features_per_sec)
    sentence_features_mat = features_tensor[features_no_start:features_no_end]
    video_id = os.path.basename(i3d_mat_file_path)
    extracted_numpy_file_path = f"{numpy_folder_path}/{video_id}_{sentence_no}.npy"
    np.save(extracted_numpy_file_path, sentence_features_mat)
    return extracted_numpy_file_path, features_no_end-features_no_start

def process_vtt_file(vtt_file_path, i3d_folder_path, npy_folder_path, tsv_file_path, features_per_sec, processed_file_count, all_file_count):
    with open(vtt_file_path, 'r') as vtt_file:
        sentence_no = 1
        video_id = os.path.splitext(os.path.basename(vtt_file_path))[0]
        for line in vtt_file:
            if line and is_time_line(line):
                next_line = next(vtt_file, '')  # Get the next line or an empty string if end of file
                bracket_pattern = r"\[(.*?)\]" # [NOT_SIGNED] or [SUB-DOES-NOT-MATCH-SIGNS] or [DUPLICATE]
                if not re.search(bracket_pattern, next_line):
                    start_sec, end_sec = extract_start_end_sec(line)
                    i3d_file_path = os.path.join(i3d_folder_path, video_id)
                    npy_file_path, sign_features_length = extract_sentence_features_matrix(i3d_file_path, start_sec, end_sec, features_per_sec, sentence_no, npy_folder_path)
                    sentence_features_id = f"{video_id}_{sentence_no}"
                    tsv_append_sentence_info(tsv_file_path, sentence_features_id, npy_file_path, sign_features_length, next_line)
                    sentence_no = sentence_no + 1
                    print(f"Appended new sign-to-sentence features data {sentence_features_id} ({processed_file_count+1}/{all_file_count} files)")
        print(f"Processed vtt file: {video_id}.vtt ")

                    
def process_vtt_folder(vtt_folder_path, i3d_folder_path, npy_folder_path, tsv_file_path, features_per_sec): 
    for root, _, files in os.walk(vtt_folder_path):
        vtt_file_count = 0
        for file_name in files:
            if file_name.endswith(".vtt"):
                vtt_file_path = os.path.join(root, file_name)
                print("Processing vtt file: ", file_name)
                process_vtt_file(vtt_file_path, i3d_folder_path, npy_folder_path, tsv_file_path, features_per_sec, vtt_file_count, len(files))
                vtt_file_count = vtt_file_count + 1
        print(f"-----------------------------------Successfully processed all {vtt_file_count} vtt file-----------------------------------")

def tsv_file_create(file_directory, file_name, column_names_list):
    tsv_file_path = f"{file_directory}/{file_name}"
    if not os.path.exists(tsv_file_path):
        with open(tsv_file_path, 'w', newline='') as tsv_file:
            writer = csv.DictWriter(tsv_file, delimiter='\t', fieldnames=column_names_list)
            writer.writeheader()
    return tsv_file_path

def tsv_append_sentence_info(tsv_file_path, id, npy_file_path, signs_length, translation_sentence, signs_offset=0, signs_type="i3d", signs_lang="bsl", translation_lang="en", topic=""):
    new_row_data = {
        "id": id,
        "signs_file": npy_file_path,
        "signs_offset": signs_offset,
        "signs_length": signs_length,
        "signs_type": signs_type,
        "signs_lang": signs_lang,
        "translation": translation_sentence,
        "translation_lang": translation_lang,
        "glosses": "",
        "topic": topic,
        "signer_id": ""
    }
    with open(tsv_file_path, mode='a', newline='', encoding='utf-8') as tsv_file:
        writer = csv.DictWriter(tsv_file, delimiter='\t', fieldnames=new_row_data.keys())
        writer.writerow(new_row_data)

def npy_folder_create(folder_directory, folder_name):
    folder_path = os.path.join(folder_directory, folder_name)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path


def main():
    with open('config.json') as config_file:
        config_data = json.load(config_file)
    I3D_FEATURES_PATH = config_data["I3D_FEATURES_PATH"]
    VTT_SUBTITLE_PATH = config_data["VTT_SUBTITLE_PATH"]
    OUTPUT_TSV_DIRECTORY = config_data["OUTPUT_TSV_DIRECTORY"]
    OUTPUT_TSV_NAME = config_data["OUTPUT_TSV_NAME"]
    OUTPUT_NPY_FOLDER_DIRECTORY = config_data["OUTPUT_NPY_FOLDER_DIRECTORY"]
    OUTPUT_NPY_FOLDER_NAME = config_data["OUTPUT_NPY_FOLDER_NAME"]
    FEATURES_PER_SEC = config_data["FEATURES_PER_SEC"]
    TSV_COLUMN_NAMES = config_data["TSV_COLUMN_NAMES"]
    npy_folder_path = npy_folder_create(OUTPUT_NPY_FOLDER_DIRECTORY, OUTPUT_NPY_FOLDER_NAME)
    tsv_file_path = tsv_file_create(OUTPUT_TSV_DIRECTORY, OUTPUT_TSV_NAME, TSV_COLUMN_NAMES)
    process_vtt_folder(VTT_SUBTITLE_PATH, I3D_FEATURES_PATH, npy_folder_path, tsv_file_path, FEATURES_PER_SEC)


if __name__ == "__main__":
    main()



