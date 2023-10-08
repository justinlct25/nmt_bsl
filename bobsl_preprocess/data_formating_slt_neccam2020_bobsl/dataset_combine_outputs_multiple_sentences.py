
import os
import json
import torch
from helpers import save_dataset_file
from dataset_splitting import split_save_dataset


## combine output from vtt process
## combine sign features and spotted signs

def concat_sentences_sign_data(sentences_list):
    sign_data_entry = {'name': '', 'signer': '', 'gloss': '', 'text': '', 'sign': ''}
    for sentence_data in sentences_list:
        sign_data_entry['name'] = sentence_data['name'] if sign_data_entry['name'] == '' else sign_data_entry['name'] + f"_{sentence_data['name']}"
        sign_data_entry['name'] = sentence_data['signer']
        sign_data_entry['gloss'] = sentence_data['gloss'] if sign_data_entry['gloss'] == '' else sign_data_entry['gloss'] + f" {sentence_data['gloss']}"
        sign_data_entry['text'] = sentence_data['text'] if sign_data_entry['text'] == '' else sign_data_entry['text'] + f" {sentence_data['text']}"
    return sign_data_entry

def is_empty_gloss(count, glosses_str):
    if glosses_str == "":
        count['empty_gloss'] += 1
        return True
    return False

def is_invalid_tensor(count, tensor):
    if not len(tensor.shape) == 2:
        count['invalid_tensor'] += 1
        return True
    elif not tensor.shape[1] == 1024 and tensor.shape[0] > 0:
        count['invalid_tensor_size'] += 1
        return True
    return False


def combine_gloss_features_multi_sentences(vtt_process_output_folder, features_extraction_model, spotting_prob, min_glosses_match=0, buffer_sec=0, max_pause_sec=0, dataset_directory='./', dataset_folder_name=None, file_limit=float('inf')):
    dataset_data = []
    count = {'sentences': 0, 'glosses': 0, 'words': 0, 'frames': 0, 'empty_gloss': 0, "invalid_tensor": 0, "invalid_tensor_size": 0, "nomatch": 0}
    features_folder = f"{vtt_process_output_folder}/vtt_process_sign_features_output_{features_extraction_model}"
    gloss_folder = f"{vtt_process_output_folder}/vtt_process_gloss_spotting_output{spotting_prob}"
    features_files_list = os.listdir(features_folder)
    features_files_num = len(features_files_list)
    file_count = 1
    for features_file in features_files_list:
        video_name = os.path.splitext(os.path.basename(features_file))[0]
        print(f"Processing sign video: {video_name} ({file_count}/{features_files_num})")
        spotting_json = os.path.join(gloss_folder, video_name)
        with open(spotting_json, 'r') as file:
            glosses_info_all = json.load(file)
        video_pth_file = os.path.join(features_folder, features_file)
        video_features_all = torch.load(video_pth_file)
        past_sentences = []
        for video_features_sentence in video_features_all:
            sentence_id = video_features_sentence['name']
            sign_tensor = video_features_sentence['sign']
            glosses_info_sentence = glosses_info_all[sentence_id]
            glosses_str = glosses_info_sentence['gloss']
            if is_invalid_tensor(count, sign_tensor):
                continue
            sentence_sign_data = {
                'name': sentence_id,
                'signer': 'Signer01',
                'gloss': glosses_str,
                'text': glosses_info_sentence['text'],
                'sign': video_features_sentence['sign']
            }
            if video_features_sentence['start'] - past_sentences[-1]['end'] > max_pause_sec:    # check the pause between current and the last sentence in the list
                sign_data_entry = concat_sentences_sign_data(past_sentences)
                past_sentences = [sentence_sign_data]
            elif video_features_sentence['end'] - past_sentences[0]['start'] > buffer_sec:  # check the total duration from the first sentence of the past sentences to the current sentence
                sign_data_entry = concat_sentences_sign_data(past_sentences)
                past_sentences = [sentence_sign_data]
            else:
                if not is_empty_gloss:
                    sign_data_entry = sentence_sign_data
                else:
                    continue
            past_sentences.append(sign_data_entry)
        file_count += 1
        if file_count>file_limit:
            break  
    dataset_info = f"Dataset:\nvideos:{features_files_num},\nratio:{split_ratios},\nsentences:{count['sentences']},\nglosses:{count['glosses']},\nwords:{count['words']},\nframes:{count['frames']},\nempty_gloss_sentence_filtered:{count['empty_gloss']},\ninvalid_tensor:{count['invalid_tensor']},\ninvalid_tensor_size:{count['invalid_tensor_size']},\ninsufficient_glosses_matches_{min_glosses_match}:{count['insufficient_glosses_matches']}"
    print(dataset_info)
    if not dataset_folder_name:
        dataset_folder_name = f"dataset_bobsl_{file_count}episodes"
    if not os.path.exists(f"{dataset_directory}/{dataset_folder_name}"):
        os.makedirs(f"{dataset_directory}/{dataset_folder_name}")
    with open(f"{dataset_directory}/{dataset_folder_name}/dataset_info.txt", "w") as file:
        file.write(dataset_info)
    # save_dataset_file(dataset_data, f"{dataset_directory}/{dataset_folder_name}/dataset.pth.gz")
    split_ratios = [0.85, 0.07, 0.08]
    split_save_dataset(dataset_data, split_ratios, dataset_folder=f"{dataset_directory}/{dataset_folder_name}")
