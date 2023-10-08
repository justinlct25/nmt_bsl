
import os
import re
import json
import torch
from helpers import is_time_line
from video2frames import sentence_video_to_frames
from frames2features import frames_to_features
from googlenet_pytorch import GoogLeNet
import concurrent.futures
import signal


# {'name': 'test/25October_2010_Monday_tagesschau-17', 'signer': 'Signer01', 'gloss': 'REGEN SCHNEE REGION VERSCHWINDEN NORD REGEN KOENNEN REGION STERN KOENNEN SEHEN', 'text': 'regen und schnee lassen an den alpen in der nacht nach im norden und nordosten fallen hier und da schauer sonst ist das klar .', 'sign': tensor([[1.2158, 0.0000, 0.4402,  ..., 0.0000, 0.0000, 0.0000], ... , [1.3268, 0.4522, 0.3599,  ..., 0.0000, 0.0000, 0.0000]])}

def process_vtt_file_sign_features(vtt_file, videos_folder, frames_folders_directory, output_folder, model, file_num=1, total_files=1):
    vtt_list = []
    video_name = os.path.splitext(os.path.basename(vtt_file))[0]
    video_file = os.path.join(videos_folder, f"{video_name}.mp4")
    with open(vtt_file, 'r') as file:
        total_sentences = sum(1 for line in file if is_time_line(line))  
    with open(vtt_file, 'r') as file:
        sentence_no = 1
        for line in file:
            if line and is_time_line(line):
                next_line = next(file, '')  # Get the next line or an empty string if end of file
                bracket_pattern = r"\[(.*?)\]" # [NOT_SIGNED] or [SUB-DOES-NOT-MATCH-SIGNS] or [DUPLICATE]
                if not re.search(bracket_pattern, next_line):
                    print(f"Processing sentence {sentence_no}/{total_sentences} of file {video_name} ({file_num}/{total_files})...") if total_files>1 else print(f"Processing sentence {sentence_no}/{total_sentences}...")
                    start_time, end_time = line.strip().split(' --> ')
                    sentence_frames_folder = f"{video_name}_{sentence_no}"
                    frames_folder = os.path.join(frames_folders_directory, sentence_frames_folder)
                    sentence_video_to_frames(video_file, frames_folder, start_time, end_time, fps=25)
                    sentence_sign_features = frames_to_features(frames_folder, model)
                    sentence_subtitle_txt = next_line.rstrip()
                    sentence_info = {
                        "name": f"{video_name}_{sentence_no}",
                        "start": start_time,
                        "end": end_time,
                        "sign": sentence_sign_features,
                        "text": sentence_subtitle_txt
                    }
                    print(f"Text: {sentence_subtitle_txt}")
                    print(f"Features: {sentence_sign_features.shape}")
                    vtt_list.append(sentence_info)
                sentence_no += 1
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    file_path = os.path.join(output_folder, f"{video_name}.pth")
    torch.save(vtt_list, file_path)


def process_vtt_folder_sign_features(vtt_folder, videos_folder, frames_folders_directory, output_folder_directory, pretrained_model, start_file_idx=1):
    model = GoogLeNet.from_pretrained(pretrained_model)
    model.eval()
    file_num = 1
    folder_files = os.listdir(vtt_folder)
    total_files = len(folder_files)
    output_folder = os.path.join(output_folder_directory, f"vtt_process_sign_features_output_googlenet")
    processed_videos = []
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        processed_videos = os.listdir(output_folder)
    for file in folder_files:
        video_name = os.path.splitext(os.path.basename(file))[0]
        if file_num >= start_file_idx and not f"{video_name}.pth" in processed_videos:
            vtt_file = os.path.join(vtt_folder, file)
            process_vtt_file_sign_features(vtt_file, videos_folder, frames_folders_directory, output_folder, model, file_num=file_num, total_files=total_files)
        file_num += 1

def interrupt_handler(signum, frame):
    print("Interrupt signal received. Stopping all processes...")
    os.killpg(0, signal.SIGINT)  # Sends the SIGINT signal to the entire process group

def process_vtt_folder_sign_features_multiprocess(vtt_folder, videos_folder, frames_folders_directory, output_folder_directory, pretrained_model, start_file_idx=1):
    model = GoogLeNet.from_pretrained(pretrained_model)
    model.eval()
    file_num = 1
    folder_files = os.listdir(vtt_folder)
    total_files = len(folder_files)
    output_folder = os.path.join(output_folder_directory, f"vtt_process_sign_features_output_googlenet")
    processed_videos = []
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        processed_videos = os.listdir(output_folder)
    signal.signal(signal.SIGINT, interrupt_handler)
    # with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:  
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = []
        for file in folder_files:
            video_name = os.path.splitext(os.path.basename(file))[0]
            if file_num >= start_file_idx and not f"{video_name}.pth" in processed_videos:
                vtt_file = os.path.join(vtt_folder, file)
                future = executor.submit(process_vtt_file_sign_features, vtt_file, videos_folder, frames_folders_directory, output_folder, model, file_num=file_num, total_files=total_files)
                futures.append(future)
                processed_videos.append(video_name)
            else:
                print(f"{video_name}.pth already exist {file_num}/{total_files}")
            file_num += 1
        concurrent.futures.wait(futures)
        


def main():
    with open('config.json') as config_file:
        config_data = json.load(config_file)
    VTT_SUBTITLE_FOLDER = config_data["VTT_SUBTITLE_FOLDER"]
    VIDEOS_FOLDER = config_data["VIDEOS_FOLDER"]
    FRAMES_FOLDER_DIRECTORY = "./subtitle_sentence_frames"
    VTT_PROCESS_FEATURES_OUTPUT_FOLDER_DIRECTORY = config_data["VTT_PROCESS_FEATURES_OUTPUT_FOLDER_DIRECTORY"]
    FEATURE_EXTRACTION_PRETRAINED_MODEL = config_data["FEATURE_EXTRACTION_PRETRAINED_MODEL"]
    # process_vtt_folder_sign_features(VTT_SUBTITLE_FOLDER, VIDEOS_FOLDER, FRAMES_FOLDER_DIRECTORY, VTT_PROCESS_FEATURES_OUTPUT_FOLDER_DIRECTORY, FEATURE_EXTRACTION_PRETRAINED_MODEL)
    process_vtt_folder_sign_features_multiprocess(VTT_SUBTITLE_FOLDER, VIDEOS_FOLDER, FRAMES_FOLDER_DIRECTORY, VTT_PROCESS_FEATURES_OUTPUT_FOLDER_DIRECTORY, FEATURE_EXTRACTION_PRETRAINED_MODEL)
    # process_vtt_folder_sign_features(VTT_SUBTITLE_FOLDER, VIDEOS_FOLDER, FRAMES_FOLDER_DIRECTORY, VTT_PROCESS_FEATURES_OUTPUT_FOLDER_DIRECTORY, FEATURE_EXTRACTION_PRETRAINED_MODEL)


if __name__ == "__main__":
    main()
