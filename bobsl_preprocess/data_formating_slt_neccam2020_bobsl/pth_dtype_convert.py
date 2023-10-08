import os
import torch

# Define the folder path containing .pth files
folder_path = '/Users/chuntingjustinlo/Study/CardiffComputing/DissertationSignify/self_bobsl_preprocess/data_formating_slt_neccam2020_bobsl/vtt_process_output/vtt_process_sign_features_output_i3d'

def convert_to_double(item):
    if isinstance(item, torch.Tensor):
        return item.double()
    return item

def convert_to_float_32(item):
    if isinstance(item, torch.Tensor):
        return item.type(torch.float32)
    return item


# Loop through the files in the folder
count = 0
files_list = os.listdir(folder_path)
for filename in files_list:
    if filename.endswith('.pth'):
        print(f"Processing file {count}/{len(files_list)}...")
        count += 1
        file_path = os.path.join(folder_path, filename)
        sentences_data = torch.load(file_path)
        for sentence_data in sentences_data:
            tensor = sentence_data['sign']
            sentence_data['sign'] = convert_to_float_32(tensor)
        
        
########## gave up
        # torch.save(sentence_data, file_path)
        
        # Save the converted data back to the .pth file
