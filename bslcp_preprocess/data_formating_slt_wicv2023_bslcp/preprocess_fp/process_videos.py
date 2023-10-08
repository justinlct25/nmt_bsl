import os
from process_eaf import load_eaf_annotations
from operations_i3d import get_video_fps, ms_to_frame, extract_sign_features
from operations_tsv import create_row_dict, append_sentence_data
import numpy as np
import signal
import concurrent.futures

video_extensions = [".mp4", ".mov"]


def get_valid_sign_videos(video_folder):
    valid_videos = []
    files_list = os.listdir(video_folder)
    for file in files_list:
        if not file.startswith('._'):
            filename = os.path.splitext(file)[0]
            is_valid_video_ext = any(file.endswith(ext) for ext in video_extensions)
            if is_valid_video_ext and f"{filename}.eaf" in files_list:
                valid_videos.append(filename)
    return valid_videos

def create_sentence_npy_file(npy_folder, video_name, sentence_idx, spatial_features):
    video_npy_folder = os.path.join(npy_folder, video_name)
    if not os.path.exists(video_npy_folder):
        os.mkdir(video_npy_folder)
    sentence_npy_file = os.path.join(video_npy_folder, f"{video_name}_{sentence_idx+1}.npy")
    np.save(sentence_npy_file, spatial_features)
    return sentence_npy_file

def process_video_eaf(videos_folder, video_name, model, npy_folder, tsv_file, video_num, total_videos, dataset_activity, is_features_per_frame=False):
    print(f"Processing video {video_name} ({video_num}/{total_videos} videos)... ")
    video_path = os.path.join(videos_folder, f"{video_name}.mov")
    eaf_path = os.path.join(videos_folder, f"{video_name}.eaf")
    sentences = load_eaf_annotations(eaf_path)
    fps = get_video_fps(video_path)
    for sentence_idx, sentence in enumerate(sentences):
        start_frame, end_frame = ms_to_frame(fps, [sentence['start'], sentence['end']])
        spatial_features = extract_sign_features(video_path, start_frame, end_frame, model, is_features_per_frame=is_features_per_frame, center_square_crop=True)
        npy_file = create_sentence_npy_file(npy_folder, video_name, sentence_idx, spatial_features)
        np.save(npy_file, spatial_features)
        tsv_row_dict = create_row_dict(f"{video_name}_{sentence_idx+1}", npy_file, int(end_frame-start_frame), sentence['value'], int(start_frame), topic=dataset_activity, fps=fps)
        append_sentence_data(tsv_file, tsv_row_dict)
        print(f"Appended new features data of sign sentence {sentence_idx+1}/{len(sentences)} of video \"{video_name}\" ({video_num}/{total_videos} videos of folder \"{dataset_activity}\")")

def process_sign_videos_folder(videos_folder, model, npy_folder, tsv_file, dataset_activity):
    video_count = 1
    processed_videos = os.listdir(npy_folder)
    valid_sign_videos = get_valid_sign_videos(videos_folder)
    for video_name in valid_sign_videos:
        if not video_name in processed_videos:
            process_video_eaf(videos_folder, video_name, model, npy_folder, tsv_file, video_count, len(valid_sign_videos), dataset_activity)
            video_count += 1
        else:
            print(f"{video_name} already processed")

def multi_process_interrupt_handler():
    print("Interrupt signal received. Stopping all processes...")
    os.killpg(0, signal.SIGINT)  # Sends the SIGINT signal to the entire process group

# def process_sign_videos_folder_multi_process(videos_folder, npy_folder, model):
#     video_count = 1
#     processed_videos = os.listdir(npy_folder)
#     valid_sign_videos = get_valid_sign_videos(videos_folder)
#     signal.signal(signal.SIGINT, multi_process_interrupt_handler)
#     with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
#         futures = []
#         for video_name in valid_sign_videos:
#             if not video_name in processed_videos:
#                 future = executor.submit(process_video_eaf, video_name, video_count, model)
#                 futures.append(future)
#                 processed_videos.append(video_name)
#             else:
#                 print(f"{video_name} already processed")
#             video_count += 1
#         concurrent.futures.wait(futures)

def process_sign_videos_folder_multi_process(videos_folder, model, npy_folder, tsv_file, dataset_activity, is_features_per_frame=False):
    video_count = 1
    valid_sign_videos = get_valid_sign_videos(videos_folder)
    signal.signal(signal.SIGINT, multi_process_interrupt_handler)
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = []
        for video_name in valid_sign_videos:
            processed_videos = os.listdir(npy_folder)
            if not video_name in processed_videos:
                future = executor.submit(process_video_eaf, videos_folder, video_name, model, npy_folder, tsv_file, video_count, len(valid_sign_videos), dataset_activity, is_features_per_frame)
                futures.append(future)
            else:
                print(f"{video_name} video {video_count}/{len(valid_sign_videos)} already processed, skip")
            video_count += 1
        concurrent.futures.wait(futures)