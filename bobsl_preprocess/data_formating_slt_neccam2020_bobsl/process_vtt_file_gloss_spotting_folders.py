
import os
import re
import json
from signs_spotting import signs_spotting
from helpers import is_time_line


def process_vtt_file(vtt_file, spotting_folder, video_file, frame_folders_path, model):
    with open(vtt_file, 'r') as file:
        total_lines = sum(1 for line in file if is_time_line(line))  
    with open(vtt_file, 'r') as file:
        sentence_no = 1
        video_id = os.path.splitext(os.path.basename(vtt_file))[0]
        for line in file:
            if line and is_time_line(line):
                next_line = next(file, '')  # Get the next line or an empty string if end of file
                bracket_pattern = r"\[(.*?)\]" # [NOT_SIGNED] or [SUB-DOES-NOT-MATCH-SIGNS] or [DUPLICATE]
                if not re.search(bracket_pattern, next_line):
                    start_time, end_time = line.strip().split(' --> ')
                    signs_str = signs_spotting(video_id, start_time, end_time, spotting_folder, 0.7, nltk=True)
                    subtitle_sentence = next_line.rstrip()
                    # sentence_frames_folder = f"{video_id}_{sentence_no}"
                    # frames_folder = os.path.join(frame_folders_path, sentence_frames_folder)
                    # sentence_video_to_frames(video_file, frames_folder, start_time, end_time, fps=25)
                    # sentence_sign_features = frames_to_features(frames_folder, model)
                    print(signs_str)
                    print(subtitle_sentence)
                    print(f"{sentence_no}/{total_lines}")
                    sentence_no = sentence_no + 1

def process_vtt_file_gloss_text(vtt_file, spotting_folder, output_folder, prob, file_num=1, total_files=1):
    vtt_dict = {}
    video_id = os.path.splitext(os.path.basename(vtt_file))[0]
    with open(vtt_file, 'r') as file:
        total_sentences = sum(1 for line in file if is_time_line(line))  
    with open(vtt_file, 'r') as file:
        sentence_no = 1
        for line in file:
            if line and is_time_line(line):
                next_line = next(file, '')  # Get the next line or an empty string if end of file
                bracket_pattern = r"\[(.*?)\]" # [NOT_SIGNED] or [SUB-DOES-NOT-MATCH-SIGNS] or [DUPLICATE]
                if not re.search(bracket_pattern, next_line):
                    print(f"Processing sentence {sentence_no}/{total_sentences} of file {file_num}/{total_files}...") if total_files>1 else print(f"Processing sentence {sentence_no}/{total_sentences}...")
                    start_time, end_time = line.strip().split(' --> ')
                    signs_list = signs_spotting(video_id, start_time, end_time, spotting_folder, prob, nltk=True)
                    signs_str = " ".join(signs_list)
                    subtitle_txt = next_line.rstrip()
                    sentence_info = {
                        "gloss": signs_str,
                        "text": subtitle_txt
                    }
                    vtt_dict[f"{video_id}_{sentence_no}"] = sentence_info
                    print(f"Gloss: {signs_str}")
                    print(f"Text: {subtitle_txt}")
                sentence_no = sentence_no + 1
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    file_path = os.path.join(output_folder, video_id)
    with open(file_path, "w") as json_file:
        json.dump(vtt_dict, json_file)

def process_vtt_folder_gloss_text(vtt_folder, spotting_folder, output_folder_directory, prob=0.7):
    file_num = 1
    folder_files = os.listdir(vtt_folder)
    total_files = len(folder_files)
    output_folder = os.path.join(output_folder_directory, f"vtt_process_gloss_spotting_output_{prob}")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for file in folder_files:
        vtt_file = os.path.join(vtt_folder, file)
        process_vtt_file_gloss_text(vtt_file, spotting_folder, output_folder, prob, file_num, total_files)
        file_num = file_num + 1

    

def vtt_file_to_sign_features(video_file, model, frame_folders_path=None):
    pass


def main():
    with open('config.json') as config_file:
        config_data = json.load(config_file)
    VTT_SUBTITLE_FOLDER = config_data["VTT_SUBTITLE_FOLDER"]
    SPOTTING_FOLDER = config_data["SPOTTING_FOLDER"]
    VTT_PROCESS_GLOSS_OUTPUT_FOLDER_DIRECTORY = config_data["VTT_PROCESS_GLOSS_OUTPUT_FOLDER_DIRECTORY"]
    process_vtt_folder_gloss_text(VTT_SUBTITLE_FOLDER, SPOTTING_FOLDER, VTT_PROCESS_GLOSS_OUTPUT_FOLDER_DIRECTORY, 0.7)

if __name__ == "__main__":
    main()
