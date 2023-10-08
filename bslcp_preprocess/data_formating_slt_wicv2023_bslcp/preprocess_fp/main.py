
from helpers import load_config
from process_dataset import create_output_folders, list_video_folders, process_video_data_folders
from operations_i3d import model_init

def main():
    config = load_config()
    dataset_folder = config["BSLCP_FOLDER"]
    video_extensions = config["VIDEO_EXTENSIONS"]
    model = model_init()
    npy_folder_path, tsv_file = create_output_folders()
    video_data_folders = list_video_folders(dataset_folder, video_extensions)
    process_video_data_folders(video_data_folders, model, npy_folder_path, tsv_file, True)

if __name__ == "__main__":
    main()
    