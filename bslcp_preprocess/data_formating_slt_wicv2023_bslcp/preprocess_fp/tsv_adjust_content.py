import pandas as pd

# Replace these paths with your input and output file paths
# input_file = "./dataset_preprocess_output/i3d.bslcp.tsv"
# output_file = "./dataset_preprocess_output/i3d.bslcp_1.tsv"

# # Replace "column_name" with the name of the specific column you want to modify
# column_name = "signs_file"

# # Read the TSV file into a DataFrame
# df = pd.read_csv(input_file, sep='\t')

# # Modify the content of the specified column by appending ".npy" to each element
# df[column_name] = df[column_name].apply(lambda x: str(x) + ".npy")

# # Write the modified DataFrame back to a TSV file
# df.to_csv(output_file, sep='\t', index=False)


import pandas as pd

# Load the TSV file into a DataFrame
df = pd.read_csv('/Users/chuntingjustinlo/Study/CardiffComputing/DissertationSignify/self_bslcp_preprocess/data_formating_slt_wicv2023_bslcp/preprocess_fp/dataset_preprocess_output/i3d.val.bslcp.tsv', sep='\t')

# Specify the column name you want to delete
column_to_delete = 'fps'

# Delete the specified column
if column_to_delete in df:
    df.drop(column_to_delete, axis=1, inplace=True)

# Save the modified DataFrame to a new TSV file
df.to_csv('/Users/chuntingjustinlo/Study/CardiffComputing/DissertationSignify/self_bslcp_preprocess/data_formating_slt_wicv2023_bslcp/preprocess_fp/dataset_preprocess_output/i3d.val.bslcp.tsv', sep='\t', index=False)
