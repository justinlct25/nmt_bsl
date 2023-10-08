
import os
import json
import torch
import gzip
import pickle
from helpers import save_dataset_file
from dataset_splitting import split_save_dataset


## combine output from vtt process
## combine sign features and spotted signs
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


def match_appeared_glosses_in_text(glosses_str, text_str, stem_comparison=False):
    appeared_glosses = []
    gloss_words = glosses_str.lower().split()
    if stem_comparison:
        stemmer = PorterStemmer()
        text_stems = [stemmer.stem(token) for token in word_tokenize(text_str)]
        for word in gloss_words:
            if stemmer.stem(word) in text_stems:
                appeared_glosses.append(word)
    else:
        text_words = text_str.lower().split()
        for word in gloss_words:
            if word in text_words:
                appeared_glosses.append(word)
    return appeared_glosses

def is_invalid_tensor(count, sign_tensor):
    filter_required = False
    if not len(sign_tensor.shape) == 2:
        count['invalid_tensor'] += 1
        filter_required = True
    elif not sign_tensor.shape[1] == 1024 and sign_tensor.shape[0] > 0:
        count['invalid_tensor_size'] += 1
        filter_required = True
    return filter_required

default_filter = {
    "sentences_empty_gloss": False, 
    "sentences_min_glosses_match": 0,
    "sentences_min_glosses_text_ratio": 0,
    "glosses_not_appeared": False,
    "min_sentence_words": 0,
    "max_sentence_words": float('inf')
}

def combine_gloss_features(vtt_process_output_folder, features_extraction_model, spotting_prob, filter=default_filter, dataset_directory='./', dataset_folder_name=None, file_limit=float('inf'), stem_comparison=False):
    dataset_data = []
    count = {'sentences': 0, 'glosses': 0, 'words': 0, 'frames': 0, 'empty_gloss': 0, "invalid_tensor": 0, "invalid_tensor_size": 0, "insufficient_glosses_matches": 0, "glosses_not_appeared_in_sub": 0, "insufficient_glosses_text_ratio": 0, "under_min_words_sentences": 0, "over_max_words_sentences": 0}
    features_folder = f"{vtt_process_output_folder}/vtt_process_sign_features_output_{features_extraction_model}"
    gloss_folder = f"{vtt_process_output_folder}/vtt_process_gloss_spotting_output_{spotting_prob}"
    features_files_list = os.listdir(features_folder)
    features_files_num = len(features_files_list)
    file_count = 1
    min_glosses_match = filter['sentences_min_glosses_match']
    min_glosses_text_ratio = filter['sentences_min_glosses_text_ratio']
    min_sentence_words = filter['min_sentence_words']
    max_sentence_words = filter['max_sentence_words']
    for features_file in features_files_list:
        video_name = os.path.splitext(os.path.basename(features_file))[0]
        print(f"Processing sign video: {video_name} ({file_count}/{features_files_num})")
        spotting_json = os.path.join(gloss_folder, video_name)
        with open(spotting_json, 'r') as file:
            glosses_info_all = json.load(file)
        video_pth_file = os.path.join(features_folder, features_file)
        video_features_all = torch.load(video_pth_file)
        for video_features_sentence in video_features_all:
            sentence_id = video_features_sentence['name']
            sign_tensor = video_features_sentence['sign']
            glosses_info_sentence = glosses_info_all[sentence_id]
            glosses_str = glosses_info_sentence['gloss']
            text_str = glosses_info_sentence['text']
            if is_invalid_tensor(count, sign_tensor):
                continue
            if filter['sentences_empty_gloss'] and glosses_str == "":
                count['empty_gloss'] += 1
                continue
            appeared_glosses = match_appeared_glosses_in_text(glosses_str, text_str, stem_comparison)
            if len(appeared_glosses) < min_glosses_match:
                count['insufficient_glosses_matches'] += 1
                continue
            if len(appeared_glosses)/len(text_str.split()) < min_glosses_text_ratio:
                count['insufficient_glosses_text_ratio'] += 1
                continue
            if len(text_str.split(" ")) < min_sentence_words:
                count['under_min_words_sentences'] += 1
                continue
            if len(text_str.split(" ")) > max_sentence_words:
                count['over_max_words_sentences'] += 1
                continue
            count['glosses_not_appeared_in_sub'] += len(glosses_str.split()) - len(appeared_glosses) if filter['glosses_not_appeared'] else 0 # for filtering out glosses that are not appear in the sentence text
            glosses_str if not filter['glosses_not_appeared'] else " ".join(appeared_glosses)
            sentence_sign_data = {
                'name': sentence_id,
                'signer': 'Signer01',
                'gloss': glosses_str if not filter['glosses_not_appeared'] else " ".join(appeared_glosses),
                'text': text_str,
                'sign': video_features_sentence['sign']
            }
            count['sentences'] += 1
            count['glosses'] += len(glosses_str.split())
            count['words'] += len(glosses_info_sentence['text'].split())
            count['frames'] += video_features_sentence['sign'].shape[0]
            dataset_data.append(sentence_sign_data)
        file_count += 1
        if file_count>file_limit:
            break  
    split_ratios = [0.85, 0.07, 0.08]
    dataset_info = f'''
    Dataset:
    videos:{features_files_num},
    model:{features_extraction_model},
    spotting_prob:{spotting_prob},
    stem_comparison:{stem_comparison},
    ratio:{split_ratios},
    sentences:{count['sentences']},
    glosses:{count['glosses']},
    words:{count['words']},
    frames:{count['frames']},
    is_filter_glosses_not_appeared_in_txt:{filter['glosses_not_appeared']}
    min_words{min_sentence_words},
    max_words:{max_sentence_words}
    min_glosses_matches:{min_glosses_match},
    min_glosses_txt_ratio:{min_glosses_text_ratio},
    glosses_not_appeared_filtered:{count['glosses_not_appeared_in_sub']},
    sentences_empty_gloss_filtered:{count['empty_gloss']},
    sentences_invalid_tensor_filtered:{count['invalid_tensor']},
    sentences_invalid_tensor_size_filtered:{count['invalid_tensor_size']},
    sentences_insufficient_glosses_matches_filtered:{count['insufficient_glosses_matches']},
    sentences_insufficient_glosses_text_ratio_filtered:{count['insufficient_glosses_text_ratio']}
    sentences_under_min_words_filtered:{count['under_min_words_sentences']},
    sentences_over_max_words_filtered:{count['over_max_words_sentences']}
'''
    print(dataset_info)
    if not dataset_folder_name:
        dataset_folder_name = f"dataset_bobsl_{features_files_num}episodes_{spotting_prob}_{features_extraction_model}{'_stemC' if stem_comparison else ''}_filtered{'_0glossS' if filter['sentences_empty_gloss'] else ''}_<{min_glosses_match}glossmatchedS_<{min_glosses_text_ratio}glosstxtratioS{'_0appearG' if filter['glosses_not_appeared'] else ''}_<{min_sentence_words}words_>{max_sentence_words if not max_sentence_words == float('inf') else '(inf)'}words_sentence({count['sentences']})_new"
    if not os.path.exists(f"{dataset_directory}/{dataset_folder_name}"):
        os.makedirs(f"{dataset_directory}/{dataset_folder_name}")
    with open(f"{dataset_directory}/{dataset_folder_name}/dataset_info.txt", "w") as file:
        file.write(dataset_info)
    # save_dataset_file(dataset_data, f"{dataset_directory}/{dataset_folder_name}/dataset.pth.gz")
    split_save_dataset(dataset_data, split_ratios, dataset_folder=f"{dataset_directory}/{dataset_folder_name}")



if __name__ == '__main__':
    # combine_gloss_features("./vtt_process_output", "googlenet", "0.7", dataset_directory="./bobsl_preprocessed_datasets", min_glosses_match=2)
    # filter = { # before 20230902
    #     "sentences_empty_gloss": True, 
    #     "sentences_min_glosses_match": 1,
    #     "sentences_min_glosses_text_ratio": 0.1,
    #     "glosses_not_appeared": True
    # }
    filter = {
        "sentences_empty_gloss": True, 
        "sentences_min_glosses_match": 1,
        "sentences_min_glosses_text_ratio": 0,
        "glosses_not_appeared": True,
        "min_sentence_words": 0,
        # "min_sentence_words": 9,
        # "max_sentence_words": float('inf'),
        "max_sentence_words": 30
    }
    # combine_gloss_features("./vtt_process_output", "i3d", "0.7", filter=filter, stem_comparison=True, dataset_directory="./bobsl_preprocessed_datasets")
    # combine_gloss_features("./vtt_process_output", "googlenet", "0.5", filter=filter, stem_comparison=True, dataset_directory="./bobsl_preprocessed_datasets")
    # combine_gloss_features("./vtt_process_output", "i3d", "0.5", filter=filter, stem_comparison=True, dataset_directory="./bobsl_preprocessed_datasets")
    combine_gloss_features("./vtt_process_output", "i3d", "0.9", filter=filter, stem_comparison=True, dataset_directory="./bobsl_preprocessed_datasets")

    