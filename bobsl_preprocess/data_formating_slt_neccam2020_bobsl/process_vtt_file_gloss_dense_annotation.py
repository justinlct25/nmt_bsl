
import os
import re
import json
import concurrent.futures
import signal
from helpers import is_time_line, load_config_dict
from signs_spotting import signs_spotting_pkl_file


def process_vtt_file_gloss_text(vtt_file, annot_file, output_folder, prob, file_num=1, total_files=1, buffer_time=0):
    vtt_dict = {}
    video_name = os.path.splitext(os.path.basename(vtt_file))[0]
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
                    signs_list = signs_spotting_pkl_file(video_name, start_time, end_time, annot_file, prob, nltk=True, buffer_time=buffer_time)
                    signs_str = " ".join(signs_list)
                    subtitle_txt = next_line.rstrip()
                    sentence_info = {
                        "start": start_time,
                        "end": end_time,
                        "gloss": signs_str,
                        "text": subtitle_txt
                    }
                    vtt_dict[f"{video_name}_{sentence_no}"] = sentence_info
                    print(f"Gloss: {signs_str}")
                    print(f"Text: {subtitle_txt}")
                sentence_no += 1
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        file_path = os.path.join(output_folder, video_name)
        with open(file_path, "w") as json_file:
            json.dump(vtt_dict, json_file)


def process_vtt_folder_gloss_text(vtt_folder, annot_file, output_folder_directory, prob=0.7, buffer_time=0, start_file_idx=1):
    file_num = 1
    folder_files = os.listdir(vtt_folder)
    total_files = len(folder_files)
    output_folder = os.path.join(output_folder_directory, f"vtt_process_gloss_spotting_output_{prob}_{buffer_time}")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for file in folder_files:
        if file_num >= start_file_idx:
            vtt_file = os.path.join(vtt_folder, file)
            process_vtt_file_gloss_text(vtt_file, annot_file, output_folder, prob, file_num, total_files, buffer_time=buffer_time)
        file_num = file_num + 1


def interrupt_handler(signum, frame):
    print("Interrupt signal received. Stopping all processes...")
    os.killpg(0, signal.SIGINT)  # Sends the SIGINT signal to the entire process group

def process_vtt_folder_gloss_text_multiprocess(vtt_folder, annot_file, output_folder_directory, prob=0.7, buffer_time=0, start_file_idx=1):
    file_num = 1
    folder_files = os.listdir(vtt_folder)
    total_files = len(folder_files)
    output_folder = os.path.join(output_folder_directory, f"vtt_process_gloss_spotting_output_{prob}_{buffer_time}")
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
            if file_num >= start_file_idx and not video_name in processed_videos:
                vtt_file = os.path.join(vtt_folder, file)
                future = executor.submit(process_vtt_file_gloss_text, vtt_file, annot_file, output_folder, prob, file_num, total_files, buffer_time=buffer_time)
                futures.append(future)
                processed_videos.append(video_name)
            file_num = file_num + 1
        concurrent.futures.wait(futures)

def main():
    config = load_config_dict()
    VTT_SUBTITLE_FOLDER = config['VTT_SUBTITLE_FOLDER']
    ANNOTATION_PKL_FILE = config['ANNOTATION_PKL_FILE']
    VTT_PROCESS_GLOSS_OUTPUT_FOLDER_DIRECTORY = config['VTT_PROCESS_GLOSS_OUTPUT_FOLDER_DIRECTORY']
    # process_vtt_folder_gloss_text(VTT_SUBTITLE_FOLDER, ANNOTATION_PKL_FILE, VTT_PROCESS_GLOSS_OUTPUT_FOLDER_DIRECTORY, prob=0.7, buffer_time=0)
    process_vtt_folder_gloss_text_multiprocess(VTT_SUBTITLE_FOLDER, ANNOTATION_PKL_FILE, VTT_PROCESS_GLOSS_OUTPUT_FOLDER_DIRECTORY, prob=0.9, buffer_time=0)


if __name__ == "__main__":
    main()