import os
import json
import pickle
from helpers import is_time_in_range
from nltk.corpus import words


def signs_spotting_json_folder(video_name, start_time, end_time, spotting_folder, probability, nltk=False):
    sign_appeared = []
    for spotting_file in os.listdir(spotting_folder):
        spotting_file_path = os.path.join(spotting_folder, spotting_file)
        with open(spotting_file_path, 'r') as file:
            spotting_dict = json.load(file)

        for task, signs_dict in spotting_dict.items():
            for sign, sign_info in signs_dict.items():
                sign_occurrences = [index for index, name in enumerate(sign_info["names"]) if video_name in name]
                if len(sign_occurrences) > 0:
                    for idx in sign_occurrences:
                        sign_occurred_time = sign_info["global_times"][idx]
                        if is_time_in_range(sign_occurred_time, start_time, end_time):
                            if sign_info["probs"][idx]>probability and sign.upper() not in sign_appeared:
                                if nltk:
                                    sign_appeared.append(sign.upper()) if sign in set(words.words()) else None
                                else:
                                    sign_appeared.append(sign.upper())
    return sign_appeared


# print(signs_spotting("5085344787448740525", "00:00:22.378", "00:00:25.082", "./spottings", 0.3))

def signs_spotting_pkl_file(video_name, start_time, end_time, annot_pkl_file, probability, nltk=False, buffer_time=0):
    with open(annot_pkl_file, "rb") as file:
        data = pickle.load(file)
    video_related_info_idx = [idx for idx, name in enumerate(data['episode_name']) if video_name in name]
    sign_within_time_idx = []
    for idx in video_related_info_idx:
        sign_within_time_idx.append(idx) if is_time_in_range(data['annot_time'][idx], start_time, end_time, buffer=buffer_time) else None
    sign_appeared = []
    for idx, sign in enumerate(data['annot_word']):
        if idx in sign_within_time_idx:
            if data['annot_prob'][idx] > probability and sign.upper() not in sign_appeared:
                if nltk:
                    sign_appeared.append(sign.upper()) if sign in set(words.words()) else None
                else:
                    sign_appeared.append(sign.upper())
    return sign_appeared



# print(signs_spotting_pkl_file("5085344787448740525", "00:00:22.378", "00:00:25.082", "./auto_dense_annotations.pkl", 0.7, buffer_time=1))
# print(signs_spotting_pkl_file("5085344787448740525", "00:00:27.472", "00:00:31.992", "./auto_dense_annotations.pkl", 0.7, buffer_time=1))










