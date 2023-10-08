import os
import torch
from datetime import datetime
from dataset_splitting import split_save_dataset

def is_invalid_tensor(sign_tensor):
    if not len(sign_tensor.shape) == 2:
        return True
    elif not sign_tensor.shape[1] == 1024 and sign_tensor.shape[0] > 0:
        return True
    return False

def cal_sec(start, end):
    start_time = datetime.strptime(start, '%H:%M:%S.%f')
    end_time = datetime.strptime(end, '%H:%M:%S.%f')
    time_difference = end_time - start_time
    return time_difference.total_seconds()

def rearrange(pth_folder, output_folder, split_ratios=[0.85, 0.07, 0.08]):
    file_list = os.listdir(pth_folder)
    data_list = []
    count = {'episodes': 0, 'sentences': 0, 'duration': 0, 'words': 0, 'features': 0, 'invalid_tensor': 0}
    for file_name in file_list:
        if file_name.endswith('.pth'):
            count['episodes'] += 1
            pth_file = os.path.join(pth_folder, file_name)
            episode_sentences = torch.load(pth_file)
            for sentence in episode_sentences:
                features_tensor = sentence['sign']
                if is_invalid_tensor(features_tensor):
                    count['invalid_tensor'] += 1
                    continue
                count['sentences'] += 1
                count['duration'] += cal_sec(sentence['start'], sentence['end'])
                count['words'] += len(sentence['text'].split(" "))
                count['features'] += features_tensor.shape[0]
                sentence_data = {
                    'name': sentence['name'],
                    'signer': 'Signer01',
                    'gloss': '',
                    'text': sentence['text'],
                    'sign': sentence['sign']
                }
                data_list.append(sentence_data)
    current_time = datetime.now()
    dataset_info = f'''
        Dataset: BOBSL,
        features: I3D,
        create_at: {current_time}
        split_ratio: {split_ratios},
        episodes: {count['episodes']},
        sentences: {count['sentences']},
        duration: {count['duration']}s,
        words: {count['words']},
        features: {count['features']},
        invalid_tensor: {count['invalid_tensor']}
        '''
    print(dataset_info)
    dataset_folder_name = f"dataset_bobsl_{count['episodes']}episodes_{count['sentences']}sentences_{current_time}"
    if not os.path.exists(f"{output_folder}/{dataset_folder_name}"):
        os.makedirs(f"{output_folder}/{dataset_folder_name}")
    with open(f"{output_folder}/{dataset_folder_name}/dataset_info.txt", "w") as file:
        file.write(dataset_info)
    split_save_dataset(data_list, split_ratios, dataset_folder=f"{output_folder}/{dataset_folder_name}", gen_cos_sim=True)

if __name__ == '__main__':
    rearrange('./vtt_process_sign_features_output_i3d', './output_datasets')



        