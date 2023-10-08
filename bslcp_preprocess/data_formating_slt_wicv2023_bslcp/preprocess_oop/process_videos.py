import os
from process_eaf import load_eaf_annotations
from operations_i3d import I3dOperator
import numpy as np
import signal
import concurrent.futures

class SignVideosFolderProcessor():

    video_extensions = [".mp4", ".mov"]

    def __init__(self, video_folder, tsv_operator, npy_folder, video_extensions=None):
        self.video_folder = video_folder
        self.video_folder_name = os.path.basename(video_folder)
        self.valid_sign_videos = self.get_valid_sign_videos()
        self.total_videos = len(self.valid_sign_videos)
        self.tsv_operator = tsv_operator
        self.npy_folder = npy_folder
        self.processed_videos = os.listdir(npy_folder)
        if video_extensions:
            self.video_extensions = video_extensions

    def get_valid_sign_videos(self):
        valid_videos = []
        files_list = os.listdir(self.video_folder)
        for file in files_list:
            if not file.startswith('._'):
                filename = os.path.splitext(file)[0]
                is_valid_video_ext = any(file.endswith(ext) for ext in self.video_extensions)
                if is_valid_video_ext and f"{filename}.eaf" in files_list:
                    valid_videos.append(filename)
        return valid_videos
    
    def create_sentence_npy_file(self, video_name, sentence_idx, spatial_features):
        video_npy_folder = os.path.join(self.npy_folder, video_name)
        if not os.path.exists(video_npy_folder):
            os.mkdir(video_npy_folder)
        sentence_npy_file = os.path.join(video_npy_folder, f"{video_name}_{sentence_idx}")
        np.save(sentence_npy_file, spatial_features)
        return sentence_npy_file

    def process_video_eaf(self, video_name, video_num):
        print(f"Processing video {video_name} ({video_num}/{self.total_videos} videos)... ")
        video_path = os.path.join(self.video_folder, f"{video_name}.mov")
        eaf_path = os.path.join(self.video_folder, f"{video_name}.eaf")
        i3d_operator = I3dOperator()
        i3d_operator.load_video_file(video_path)
        sentences = load_eaf_annotations(eaf_path)
        for sentence_idx, sentence in enumerate(sentences):
            start_frame, end_frame = i3d_operator.ms_to_frame([sentence['start'], sentence['end']])
            spatial_features = i3d_operator.extract_sign_features(start_frame, end_frame)
            npy_file = self.create_sentence_npy_file(video_name, sentence_idx, spatial_features)
            np.save(self.npy_folder, spatial_features)
            tsv_row_dict = self.tsv_operator.create_row_dict(f"{video_name}_{sentence_idx}", npy_file, int(end_frame-start_frame), sentence['value'], int(start_frame), topic=self.video_folder_name, fps=i3d_operator.fps)
            self.tsv_operator.append_sentence_data(tsv_row_dict)
            print(f"Appended new features data of sign sentence {sentence_idx+1}/{len(sentences)} of video \"{video_name}\" ({video_num}/{self.total_videos} videos of folder \"{self.video_folder_name}\")")

    def process_sign_videos_folder(self):
        video_count = 1
        for video_name in self.valid_sign_videos:
            if not video_name in self.processed_videos:
                self.process_video_eaf(video_name, video_count)
                video_count += 1
            else:
                print(f"{video_name} already processed")



    def multi_process_interrupt_handler(self):
        print("Interrupt signal received. Stopping all processes...")
        os.killpg(0, signal.SIGINT)  # Sends the SIGINT signal to the entire process group

    def process_sign_videos_folder_multi_process(self):
        signal.signal(signal.SIGINT, self.multi_process_interrupt_handler)
        video_count = 1
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            futures = []
            for video_name in self.valid_sign_videos:
                if not video_name in self.processed_videos:
                    future = executor.submit(self.process_video_eaf, video_name, video_count)
                    futures.append(future)
                    self.processed_videos.append(video_name)
                else:
                    print(f"{video_name} already processed")
                video_count += 1
            concurrent.futures.wait(futures)


