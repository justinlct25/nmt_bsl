import pandas as pd
import re

# Read the TSV file into a DataFrame
tsv_file = '/Users/chuntingjustinlo/Study/CardiffComputing/DissertationSignify/self_bslcp_preprocess/data_formating_slt_neccam2020_bslcp/dataset_preprocess_output_interview_narrative_featureperframe_20230818/i3d.bslcp.original.tsv'  # Replace with your file's path
column_name = 'translation'  # Replace with the column's name

# Read the TSV file into a DataFrame
df = pd.read_csv(tsv_file, sep='\t')

# Define a function to check if a sentence is valid
def is_valid_sentence(value):
    if value is None:
        return False
    value = str(value).strip()
    return bool(re.search(r'\w', value))  # Returns True if there's at least one word character

# Filter out rows where the sentence is invalid in the specified column
filtered_df = df[df[column_name].apply(is_valid_sentence)]

# Get the filtered out rows for printing
filtered_out_rows = df[~df.index.isin(filtered_df.index)]

# Count the number of filtered rows
filtered_rows = df.shape[0] - filtered_df.shape[0]

# Write the filtered DataFrame back to a TSV file
filtered_tsv_file = '/Users/chuntingjustinlo/Study/CardiffComputing/DissertationSignify/self_bslcp_preprocess/data_formating_slt_neccam2020_bslcp/dataset_preprocess_output_interview_narrative_featureperframe_20230818/i3d.bslcp.filtered.tsv'  # Replace with your desired output file path
filtered_df.to_csv(filtered_tsv_file, sep='\t', index=False)

# Print the values of the filtered out rows
print("Values of filtered out rows:")
for index, row in filtered_out_rows.iterrows():
    print(row[column_name])

print(f"Number of rows before filtering: {df.shape[0]}")
print(f"Number of filtered rows: {filtered_rows}")
