import os
import random
from helpers import load_dataset_file, save_dataset_file
import json
from cos_sim import calculate_cos_sim
import pickle

def split_dataset(data_list, split_ratios):
    total_samples = len(data_list)
    train_size = int(total_samples * split_ratios[0])
    dev_size = int(total_samples * split_ratios[1])

    video_sentence_list, original_video_id_dict = gen_video_sentence_list_n_id_dict(data_list)
    
    # Shuffle the indices of the data
    shuffled_indices = list(range(total_samples))
    random.shuffle(shuffled_indices)
    
    train_indices = shuffled_indices[:train_size]
    dev_indices = shuffled_indices[train_size:train_size + dev_size]
    test_indices = shuffled_indices[train_size + dev_size:]

    output_video_id_dict = {}

    train_data, dev_data, test_data = [], [], []
    for i in train_indices:
        data = data_list[i]
        splitted_sentence_name = 'train/' + data['name']
        output_video_id_dict[splitted_sentence_name] = original_video_id_dict[data['name']]
        data['name'] = splitted_sentence_name
        train_data.append(data)
    for i in dev_indices:
        data = data_list[i]
        splitted_sentence_name = 'dev/' + data['name']
        output_video_id_dict[splitted_sentence_name] = original_video_id_dict[data['name']]
        data['name'] = splitted_sentence_name
        dev_data.append(data)
    for i in test_indices:
        data = data_list[i]
        splitted_sentence_name = 'test/' + data['name']
        output_video_id_dict[splitted_sentence_name] = original_video_id_dict[data['name']]
        data['name'] = splitted_sentence_name
        test_data.append(data)
    
    return train_data, dev_data, test_data, video_sentence_list, output_video_id_dict

def split_save_dataset(data_list, split_ratios, dataset_folder="./", dataset_name="dataset.pth.gz", gen_cos_sim=False):
    train_data, dev_data, test_data, video_sentence_list, video_id_dict = split_dataset(data_list, split_ratios)
    if gen_cos_sim:
        print("Calculating Cosine Similarity...")
        cos_sim_tensor = calculate_cos_sim(video_sentence_list)
        with open(f"{dataset_folder}/cos_sim.pkl", 'wb') as pkl_file:
            pickle.dump(cos_sim_tensor, pkl_file)
        with open(f"{dataset_folder}/name_to_video_id.json", "w") as json_file:
            json.dump(video_id_dict, json_file)
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
    train_data, dev_data, test_data, _, _ = split_dataset(dataset, split_ratios)
    print(f"Saving {dataset_name}.train")
    save_dataset_file(train_data, f"{dataset_name}.train")
    print(f"Saving {dataset_name}.dev")
    save_dataset_file(dev_data, f"{dataset_name}.dev")
    print(f"Saving {dataset_name}.test")
    save_dataset_file(test_data, f"{dataset_name}.test")
    print("Dataset splitting and saving completed")


def gen_video_sentence_list_n_id_dict(data_list):
    video_sentence_list = []
    video_id_dict = {}
    for i, data in enumerate(data_list):
        video_sentence_list.append(data['text'])
        video_id_dict[data['name']] = i
    return video_sentence_list, video_id_dict
