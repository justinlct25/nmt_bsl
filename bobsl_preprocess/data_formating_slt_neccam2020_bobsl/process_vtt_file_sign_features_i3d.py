
import os
import re
from helpers import load_config_dict, is_time_line
from features_from_mat_file import extract_sentence_features_tensor
import torch
import concurrent.futures
import signal


def process_vtt_file_sign_features(vtt_file, i3d_features_folder, output_folder, file_num=1, total_files=1):
    vtt_list = []
    video_name = os.path.splitext(os.path.basename(vtt_file))[0]
    video_i3d_file = os.path.join(i3d_features_folder, f"{video_name}.mat")
    with open(vtt_file, 'r') as file:
        total_sentences = sum(1 for line in file if is_time_line(line))  
    with open(vtt_file, 'r') as file:
        sentence_no = 1
        for line in file:
            if line and is_time_line(line):
                next_line = next(file, '')  # Get the next line or an empty string if end of file
                bracket_pattern = r"\[(.*?)\]" # [NOT_SIGNED] or [SUB-DOES-NOT-MATCH-SIGNS] or [DUPLICATE]
                if not re.search(bracket_pattern, next_line):
                    sentence_name = f"{video_name}_{sentence_no}"
                    print(f"Processing {sentence_name} {sentence_no}/{total_sentences} sentences of file {file_num}/{total_files}...") if total_files>1 else print(f"Processing {sentence_name} {sentence_no}/{total_sentences} sentences...")
                    start_time, end_time = line.strip().split(' --> ')
                    sentence_i3d_features = extract_sentence_features_tensor(video_i3d_file, start_time, end_time, 6.25, sentence_no) ## tensor.double()
                    sentence_subtitle_txt = next_line.rstrip()
                    sentence_info = {
                        "name": sentence_name,
                        "start": start_time,
                        "end": end_time,
                        "sign": sentence_i3d_features,
                        "text": sentence_subtitle_txt
                    }
                    vtt_list.append(sentence_info)
                sentence_no += 1
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    file_path = os.path.join(output_folder, f"{video_name}.pth")
    torch.save(vtt_list, file_path)

def process_vtt_folder_sign_features(vtt_folder, i3d_features_folder, output_folder_directory, start_file_idx=1):
    file_num = 1
    folder_files = os.listdir(vtt_folder)
    total_files = len(folder_files)
    output_folder = os.path.join(output_folder_directory, f"vtt_process_sign_features_output_i3d")
    processed_videos = []
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        processed_videos = os.listdir(output_folder)
    for file in folder_files:
        video_name = os.path.splitext(os.path.basename(file))[0]
        if file_num >= start_file_idx and not video_name in processed_videos:
            vtt_file = os.path.join(vtt_folder, file)
            process_vtt_file_sign_features(vtt_file, i3d_features_folder, output_folder, file_num=file_num, total_files=total_files)
            processed_videos.append(video_name)
        file_num += 1

def interrupt_handler():
    print("Interrupt signal received. Stopping all processes...")
    os.killpg(0, signal.SIGINT)  # Sends the SIGINT signal to the entire process group


def process_vtt_folder_sign_features_multiprocess(vtt_folder, i3d_features_folder, output_folder_directory, start_file_idx=1):
    file_num = 1
    folder_files = os.listdir(vtt_folder)
    total_files = len(folder_files)
    output_folder = os.path.join(output_folder_directory, f"vtt_process_sign_features_output_i3d")
    processed_videos = []
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        processed_videos = os.listdir(output_folder)
    signal.signal(signal.SIGINT, interrupt_handler)
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = []
        for file in folder_files:
            video_name = os.path.splitext(os.path.basename(file))[0]
            if file_num >= start_file_idx and not video_name in processed_videos:
                vtt_file = os.path.join(vtt_folder, file)
                future = executor.submit(process_vtt_file_sign_features, vtt_file, i3d_features_folder, output_folder, file_num=file_num, total_files=total_files)
                futures.append(future)
                processed_videos.append(video_name)
            file_num += 1
        concurrent.futures.wait(futures)



def main():
    config = load_config_dict()
    VTT_SUBTITLE_FOLDER = config["VTT_SUBTITLE_FOLDER"]
    VTT_PROCESS_FEATURES_OUTPUT_FOLDER_DIRECTORY = config["VTT_PROCESS_FEATURES_OUTPUT_FOLDER_DIRECTORY"]
    BOBSL_I3D_FEATURES_FOLDER = config["BOBSL_I3D_FEATURES_FOLDER"]
    # process_vtt_folder_sign_features(VTT_SUBTITLE_FOLDER, BOBSL_I3D_FEATURES_FOLDER, VTT_PROCESS_FEATURES_OUTPUT_FOLDER_DIRECTORY)
    process_vtt_folder_sign_features_multiprocess(VTT_SUBTITLE_FOLDER, BOBSL_I3D_FEATURES_FOLDER, VTT_PROCESS_FEATURES_OUTPUT_FOLDER_DIRECTORY)




if __name__ == "__main__":
    main()