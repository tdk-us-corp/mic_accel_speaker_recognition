#
# Copyright (c) [2024] TDK U.S.A. Corporation
#
import os
import random

# Setting a random seed for reproducibility
random.seed(1886)

def create_speaker_dict(root_folder):
    """
    Create a dictionary with speaker folder names as keys and a list of .wav file paths as values.

    Parameters:
    - root_folder: Path to the root folder containing the speaker folders.

    Returns:
    - A dictionary with speaker names as keys and lists of .wav file paths as values.
    """
    speaker_dict = {}
    
    # Iterate through all items in the root folder
    for item in os.listdir(root_folder):
        item_path = os.path.join(root_folder, item)
        
        # Check if the item is a directory (i.e., a speaker folder)
        if os.path.isdir(item_path):
            speaker_dict[item] = []
            
            # Iterate through all files in the speaker folder
            for file in os.listdir(item_path):
                cleaned_file = file.strip()  # Remove leading and trailing spaces
                if cleaned_file.endswith('.wav'):
                    # Add the cleaned .wav file path to the speaker's list
                    speaker_dict[item].append(os.path.join(item_path, cleaned_file))
    
    return speaker_dict

# Set the path to your data folder
data_folder = "/mnt/009-Audio/Internships/AccelAuthentification/chinese_accel_micro_databases/extracted_segments/Mic_train"

# Create the dictionary of data
data_dict = create_speaker_dict(data_folder)
print("Found Speakers: ", list(data_dict.keys()))

# Initialize dictionaries for positive and negative samples
positive_samples = {}
negative_samples = {}

# Create random positive pairs
for n, speaker in enumerate(list(data_dict.keys())):
    range_start = 0
    range_end = len(data_dict[speaker]) 
    num_of_examples = 3
    all_possible_pairs = [(i, j) for i in range(range_start, range_end) for j in range(range_start, range_end) if i != j]
    unique_pairs = random.sample(all_possible_pairs, num_of_examples)
    
    positive_samples[n] = unique_pairs

# Create negative pairs
for i, speaker in enumerate(list(data_dict.keys())):
    range_start = 0
    range_end = len(data_dict[speaker])
    num_of_speakers = len(list(data_dict.keys()))
    other_speakers = list(set(range(range_start, num_of_speakers)) - {i})
    
    chosen_speakers = random.sample(other_speakers, num_of_speakers-1-150)
    num_per_speaker = 1

    for spkr in chosen_speakers:
        spkr_key = list(data_dict.keys())[spkr]
        all_possible_pairs = [(i, j) for i in range(range_start, range_end) for j in range(range_start, len(data_dict[spkr_key])) if i != j]
        unique_pairs = random.sample(all_possible_pairs, num_per_speaker)
        negative_samples[(i, spkr)] = unique_pairs

# Open a text file for writing the verification pairs
with open('train_verification_pairs.txt', 'w') as file:
    # Write positive samples to file
    for key in list(positive_samples.keys()):
        for n1, n2 in positive_samples[key]:
            speaker = list(data_dict.keys())[key]
            p1, p2 = data_dict[speaker][n1], data_dict[speaker][n2]
            print(f"1 {os.path.abspath(p1)} {os.path.abspath(p2)}\n")
            file.write(f"1 {os.path.abspath(p1)} {os.path.abspath(p2)}\n")

    # Write negative samples to file
    for key in negative_samples.keys():
        for i, j in negative_samples[key]:
            sp1, sp2 = key
            sp1, sp2 = list(data_dict.keys())[sp1], list(data_dict.keys())[sp2]
            p1, p2 = data_dict[sp1][i], data_dict[sp2][j]
            print(f"0 {os.path.abspath(p1)} {os.path.abspath(p2)}\n")

            file.write(f"0 {os.path.abspath(p1)} {os.path.abspath(p2)}\n")

print("Verification Pairs Successfully Created")
