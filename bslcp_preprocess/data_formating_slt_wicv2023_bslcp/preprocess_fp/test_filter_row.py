import csv

# Open the original TSV file for reading
input_tsv_file_path = '/Users/chuntingjustinlo/Study/CardiffComputing/DissertationSignify/self_bslcp_preprocess/data_formating_slt_wicv2023_bslcp/preprocess_fp/dataset_preprocess_output_interview_narrative_featureperframe_20230818/i3d.bslcp.tsv'
output_tsv_file_path = '/Users/chuntingjustinlo/Study/CardiffComputing/DissertationSignify/self_bslcp_preprocess/data_formating_slt_wicv2023_bslcp/preprocess_fp/dataset_preprocess_output_interview_narrative_featureperframe_20230818/i3d.bslcp.filtered.tsv'

filtered_rows = []

with open(input_tsv_file_path, 'r', newline='') as tsvfile:
    tsv_reader = csv.DictReader(tsvfile, delimiter='\t')
    
    # Check the header row for the 'translation' column
    if 'translation' not in tsv_reader.fieldnames:
        print("No 'translation' column found in the header.")
        exit()

    # Iterate through each row
    for row in tsv_reader:
        # Check if the 'translation' column has a value
        if row['translation'].strip():
            filtered_rows.append(row)

# Write the filtered rows to a new TSV file
with open(output_tsv_file_path, 'w', newline='') as outfile:
    fieldnames = filtered_rows[0].keys()  # Use keys from the first row as fieldnames
    tsv_writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter='\t')
    tsv_writer.writeheader()
    tsv_writer.writerows(filtered_rows)

print("Filtered rows saved to", output_tsv_file_path)
