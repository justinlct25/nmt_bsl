import os
import numpy as np
import torch
from dataset_splitting import split_save_dataset




def npy_file_to_tensor(npy_file_path):
    video_features_np = np.load(npy_file_path)
    sentence_features_tensor = torch.tensor(video_features_np, dtype=torch.float32)
    return sentence_features_tensor

def rearrange(tsv_file_path, dataset_directory="./", split_ratios=[0.85, 0.07, 0.08], min_words=0, gen_cos_sim=False):
    dataset_list = []
    count = {'features':0, 'frames':0, 'words':0, 'sentences':0, 'sentences_filtered':0, 'empty_tensor':0, 'tsv_invalid_lines':[]}
    with open(tsv_file_path, 'r') as file:
        total_lines = sum(1 for _ in file)
    with open(tsv_file_path, "r") as tsv_file:
        header = tsv_file.readline().strip().split("\t")
        print(f"Start processing {total_lines} rows...")
        for line_no, line in enumerate(tsv_file):
            fields = line.strip().split("\t")
            if len(fields) == len(header):
                sentence_id = fields[header.index("id")]
                translation_txt = fields[header.index("translation")]
                if not len(translation_txt.split()) >= min_words:
                    count['sentences_filtered'] += 1
                    continue
                sign_features_tensor = npy_file_to_tensor(fields[header.index("signs_file")])
                if sign_features_tensor.shape[0] < 1:
                    count['empty_tensor'] += 1
                    continue
                sentence_sign_data = {
                    'name': sentence_id,
                    'signer': 'Signer01',
                    'gloss': '',
                    'text': translation_txt,
                    'sign': sign_features_tensor
                }
                count['sentences'] += 1
                count['words'] += len(translation_txt.split())
                count['features'] += sign_features_tensor.shape[0]
                count['frames'] += int(fields[header.index("signs_length")])
                dataset_list.append(sentence_sign_data)
                print(f"Appended sentence sign data {sentence_id} ({line_no}/{total_lines} rows)", end="\r")
            else:
                # print(f"\nInvalid tsv file line: {line_no}")
                count['tsv_invalid_lines'].append(line_no)
    print()
    dataset_folder_name = f"dataset_bslcp_{count['sentences']}sentences_>{min_words-1}words"
    dataset_info = f'''
    Dataset:
    sentences: {count['sentences']},
    words: {count['words']},
    features: {count['features']},
    frames: {count['frames']},
    min_words: {min_words},
    sentences_filtered: {count['sentences_filtered']},
    empty_tensor: {count['empty_tensor']},
    tsv_invalid_lines: {', '.join(map(str, count['tsv_invalid_lines']))},
    gen_cos_sim: {gen_cos_sim}
    '''
    dataset_folder_path = f"{dataset_directory}/{dataset_folder_name}"
    if not os.path.exists(dataset_folder_path):
        os.makedirs(dataset_folder_path)
    with open(f"{dataset_folder_path}/dataset_info.txt", "w") as file:
        file.write(dataset_info)
    print("Splitting and saving dataset...")
    split_save_dataset(dataset_list, split_ratios, dataset_folder=dataset_folder_path, gen_cos_sim=gen_cos_sim)



def main():
    # Open the TSV file for reading
    tsv_file_path = "./dataset_preprocess_output_wicv2023/i3d.bslcp.tsv"
    dataset_directory = './dataset_preproccess_output_gaslt2023'
    # rearrange(tsv_file_path, dataset_directory=dataset_directory, min_words=9, gen_cos_sim=True)
    rearrange(tsv_file_path, dataset_directory=dataset_directory, min_words=5, gen_cos_sim=True)

if __name__ == '__main__':
    main()
    