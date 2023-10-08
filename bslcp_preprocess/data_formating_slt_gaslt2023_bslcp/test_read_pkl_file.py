import pickle

# Specify the path to your .pkl file
# file_path = 'cos_sim.pkl'

# file_path = '/Users/chuntingjustinlo/Study/CardiffComputing/DissertationSignify/self_bslcp_preprocess/data_formating_slt_gaslt2023_bslcp/dataset_preproccess_output_gaslt2023/dataset_bslcp_12609sentences_>6words/cos_sim.pkl'
file_path = '/Users/chuntingjustinlo/Study/CardiffComputing/DissertationSignify/self_bslcp_preprocess/data_formating_slt_gaslt2023_bslcp/dataset_preproccess_output_gaslt2023/dataset_bslcp_18179sentences_>6words/cos_sim.pkl'

# Open the .pkl file for reading in binary mode
with open(file_path, 'rb') as file:
    # Use the pickle.load() function to deserialize the object
    loaded_object = pickle.load(file)

# Now, 'loaded_object' contains the Python object stored in the .pkl file

print(loaded_object)
print(loaded_object.shape)