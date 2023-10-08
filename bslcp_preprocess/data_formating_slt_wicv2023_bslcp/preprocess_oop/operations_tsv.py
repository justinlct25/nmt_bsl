
import os
import csv

def tsv_file_create(file_path, column_names_list):
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='') as tsv_file:
            writer = csv.DictWriter(tsv_file, delimiter='\t', fieldnames=column_names_list)
            writer.writeheader()
        print(f"Created tsv file: {file_path}")
    else:
        print(f"Using an existing tsv file: {file_path}")


class TsvOperator():

    T_tsv_row = {
        "id": None,
        "signs_file": None,
        "signs_offset": None,
        "signs_length": None,
        "signs_type": None,
        "signs_lang": None,
        "translation": None,
        "translation_lang": None,
        "glosses": None,
        "topic": None,
        "signer_id": None,
        "fps": None
    }

    def __init__(self, tsv_file_path):
        self.tsv_file_path = tsv_file_path
        self.column_names_list = self.T_tsv_row.keys()
        tsv_file_create(self.tsv_file_path, self.column_names_list)

    def append_sentence_data(self, tsv_row_dict):
        with open(self.tsv_file_path, mode='a', newline='', encoding='utf-8') as tsv_file:
            writer = csv.DictWriter(tsv_file, delimiter='\t', fieldnames=self.column_names_list)
            writer.writerow(tsv_row_dict)
    
    def create_row_dict(self, sentence_id, npy_file_path, signs_length, translation_sentence, signs_offset=0, signs_type="i3d", signs_lang="bsl", translation_lang="en", glosses="", topic="", signer_id="", fps=None):
        return {
            "id": sentence_id,
            "signs_file": npy_file_path,
            "signs_offset": signs_offset,
            "signs_length": signs_length,
            "signs_type": signs_type,
            "signs_lang": signs_lang,
            "translation": translation_sentence,
            "translation_lang": translation_lang,
            "glosses": glosses,
            "topic": topic,
            "signer_id": signer_id,
            "fps": fps
        }

