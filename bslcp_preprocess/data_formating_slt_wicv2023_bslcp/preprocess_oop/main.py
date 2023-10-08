
from helpers import load_config
from process_dataset import DatasetProcessor

def main():
    config = load_config()
    dataset_folder = config["BSLCP_FOLDER"]
    dataset_processor = DatasetProcessor(dataset_folder, multi_process=False)
    dataset_processor.process_data_folder()


if __name__ == "__main__":
    main()
    