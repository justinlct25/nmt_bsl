import torch

# Load the data from the .pth file
# loaded_data = torch.load('./vtt_process_output/vtt_process_sign_features_output/6242043045785611181.pth')
# loaded_data = torch.load('./vtt_process_output/vtt_process_sign_features_output_i3d/6242043045785611181.pth')
loaded_data = torch.load('/Users/chuntingjustinlo/Study/CardiffComputing/DissertationSignify/self_bobsl_preprocess/data_formating_slt_neccam2020_bobsl/vtt_process_output/vtt_process_sign_features_output_i3d/5952613360146027617.pth')

# Now you can work with the loaded data
for item in loaded_data:
    name = item['name']
    sign = item['sign']
    # print(f"Name: {name}")
    # print(f"Tensor shape: {sign.shape}")
    # print(f"Tensor:\n{sign}")
    # print("\n")


tensor = loaded_data[1]['sign']
# tensor = tensor.type(torch.float32)

# print(loaded_data[1]['sign'])
# print(loaded_data[1]['sign'].dtype)
torch.set_printoptions(sci_mode=False)
print(tensor)
print(tensor.dtype)
print(tensor.shape)