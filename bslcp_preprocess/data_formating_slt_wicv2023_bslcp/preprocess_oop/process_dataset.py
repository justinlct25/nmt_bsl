import os
from process_videos import SignVideosFolderProcessor
from operations_tsv import TsvOperator

class DatasetProcessor():

    video_extensions = [".mp4", ".mov"]

    def __init__(self, dataset_folder, output_folder="./dataset_process_output", tsv_file_name="i3d.bslcp.tsv", npy_folder_name="sentence_sign_i3d_features", video_extension=None, multi_process=False):
        self.dataset_folder = dataset_folder
        self.output_folder = output_folder
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        self.videos_folders_list = self.get_video_folders()
        if video_extension:
            self.video_extensions = video_extension
        self.tsv_operator = TsvOperator(os.path.join(output_folder, tsv_file_name))
        self.npy_folder = self.create_npy_folder(os.path.join(output_folder, npy_folder_name))
        self.multi_process = multi_process

    def get_video_folders(self):
        video_folders = []
        for root, dirs, files in os.walk(self.dataset_folder):
            if any(file.endswith(tuple(self.video_extensions)) for file in files):
                video_folders.append(root)
        return video_folders

    def process_data_folder(self):
        for videos_folder in self.videos_folders_list:
            videos_folder_processor = SignVideosFolderProcessor(videos_folder, self.tsv_operator, self.npy_folder)
            if self.multi_process:
                videos_folder_processor.process_sign_videos_folder_multi_process()
            else:
                videos_folder_processor.process_sign_videos_folder()


    def create_npy_folder(self, folder_path):
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        return folder_path
