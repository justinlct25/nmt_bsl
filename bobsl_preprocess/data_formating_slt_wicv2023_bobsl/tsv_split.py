import pandas as pd

# Load the TSV file into a pandas DataFrame
file_path = "i3d.bobsl.tsv"
df = pd.read_csv(file_path, sep="\t")

# Randomly shuffle the rows of the DataFrame
df_shuffled = df.sample(frac=1, random_state=42)  # Set a fixed random_state for reproducibility

# Define the split ratios, according to wic2023 how2sign splitting
train_ratio = 0.696
val_ratio = 0.039
test_ratio = 0.056

# Calculate the number of rows for each split
total_rows = len(df_shuffled)
train_rows = int(total_rows * train_ratio)
val_rows = int(total_rows * val_ratio)
test_rows = int(total_rows * test_ratio)

# Split the DataFrame into train, val, and test DataFrames
train_df = df_shuffled[:train_rows]
val_df = df_shuffled[train_rows:train_rows + val_rows]
test_df = df_shuffled[train_rows + val_rows:train_rows + val_rows + test_rows]

# Sort the DataFrames if you have a specific column to sort by (replace 'column_name' with the actual column name)
train_df = train_df.sort_values(by='id', ascending=True)
val_df = val_df.sort_values(by='id', ascending=True)
test_df = test_df.sort_values(by='id', ascending=True)

# Save each split into separate TSV files
train_df.to_csv("i3d.train.bobsl.tsv", sep="\t", index=False)
val_df.to_csv("i3d.val.bobsl.tsv", sep="\t", index=False)
test_df.to_csv("i3d.test.bobsl.tsv", sep="\t", index=False)
