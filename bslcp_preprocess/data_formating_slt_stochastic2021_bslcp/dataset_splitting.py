import os
import random
from helpers import load_dataset_file, save_dataset_file

def split_dataset(data_list, split_ratios):
    total_samples = len(data_list)
    train_size = int(total_samples * split_ratios[0])
    dev_size = int(total_samples * split_ratios[1])
    
    # Shuffle the indices of the data
    shuffled_indices = list(range(total_samples))
    random.shuffle(shuffled_indices)
    
    train_indices = shuffled_indices[:train_size]
    dev_indices = shuffled_indices[train_size:train_size + dev_size]
    test_indices = shuffled_indices[train_size + dev_size:]
    
    train_data = [data_list[i] for i in train_indices]
    dev_data = [data_list[i] for i in dev_indices]
    test_data = [data_list[i] for i in test_indices]
    
    return train_data, dev_data, test_data

def split_save_dataset(data_list, split_ratios, dataset_folder="./", dataset_name="dataset.pth.gz"):
    train_data, dev_data, test_data = split_dataset(data_list, split_ratios)
    print(f"Saving {dataset_name}.dev...")
    save_dataset_file(dev_data, f"{dataset_folder}/{dataset_name}.dev")
    print(f"Saving {dataset_name}.test...")
    save_dataset_file(test_data, f"{dataset_folder}/{dataset_name}.test")
    print(f"Saving {dataset_name}.train...")
    save_dataset_file(train_data, f"{dataset_folder}/{dataset_name}.train")
    print("Dataset splitting and saving completed")


def load_split_save_dataset(full_dataset_path, split_ratios):
    dataset = load_dataset_file(full_dataset_path)
    dataset_name = os.path.basename(dataset)
    print("Dataset loaded. Splitting dataset...")
    train_data, dev_data, test_data = split_dataset(dataset, split_ratios)
    print(f"Saving {dataset_name}.train")
    save_dataset_file(train_data, f"{dataset_name}.train")
    print(f"Saving {dataset_name}.dev")
    save_dataset_file(dev_data, f"{dataset_name}.dev")
    print(f"Saving {dataset_name}.test")
    save_dataset_file(test_data, f"{dataset_name}.test")
    print("Dataset splitting and saving completed")


def main():
    dataset_path = "/Users/chuntingjustinlo/Study/CardiffComputing/DissertationSignify/self_bobsl_preprocess/data_formating_slt_neccam2020_bobsl/bobsl_preprocessed_datasets/dataset_bobsl_56episodes/dataset.pth.gz"
    split_ratios = [0.85, 0.07, 0.08]
    load_split_save_dataset(dataset_path, split_ratios)


if __name__ == '__main__':
    main()