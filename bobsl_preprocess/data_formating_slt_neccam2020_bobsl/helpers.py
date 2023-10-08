import re
from datetime import datetime
import json
import gzip
import pickle


def load_config_dict(file='config.json'):
    with open(file) as f:
        return json.load(f)

def is_time_line(line):
    time_pattern = r"\d{2}:\d{2}:\d{2}\.\d{3}"
    return re.match(time_pattern, line)

def time_to_sec(time_str):
    time_format = "%H:%M:%S.%f"
    time_obj = datetime.strptime(time_str, time_format)
    total_seconds = (time_obj.hour * 3600) + (time_obj.minute * 60) + time_obj.second + (time_obj.microsecond / 1e6)
    return total_seconds

def is_time_in_range(time, start_time, end_time):
    return time_to_sec(start_time) <= time <= time_to_sec(end_time)

def save_dataset_file(data_list, file_path): # to serialize and compress the data into a binary file
    with gzip.open(file_path, "wb") as f: # the file is being opened in binary mode for writing
        pickle.dump(data_list, f, protocol=pickle.HIGHEST_PROTOCOL) # serialize the whole list and save it as binary
        # data_len = len(data_list)
        # for idx, item in enumerate(data_list):
        #     pickle.dump(item, f, protocol=pickle.HIGHEST_PROTOCOL) # to serialize the data and save it as binary
        #     print(f"Saving {idx + 1}/{data_len} sentences", end='\r')  # Print progress on the same line
        print(f"Saved dataset at {file_path}")

def load_dataset_file(file_path): # to decompress and deserialize the binary file
    with gzip.open(file_path, "rb") as f: # the file is being opened in binary mode for reading
        loaded_object = pickle.load(f) # to deserialize the binary data 
        return loaded_object