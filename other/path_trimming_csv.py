import pandas as pd
import os
# Load the CSV file
input_file = '/mnt/009-Audio/Internships/AccelAuthentification/chinese_accel_micro_databases/new_augment/split1/mic/fused_dev.csv'  # Replace with your actual input file name
df = pd.read_csv(input_file)

# Function to trim the paths
def trim_path(path):
    parts = path.split("/")

    # print(parts)

    new_path = ''
    for item in parts[-3:]:
        new_path = os.path.join(new_path, item)
        # print('\n',  new_path)
    print(new_path)

    return new_path

# Apply the function to the 'wav' column
df['wav'] = df['wav'].apply(lambda x: ", ".join([trim_path(p) for p in eval(x)]))

# Save to a new CSV file
output_file = 'trimmed_paths.csv'  # Replace with your desired output file name
df.to_csv(output_file, index=False)

print(f"Trimmed paths saved to {output_file}")
