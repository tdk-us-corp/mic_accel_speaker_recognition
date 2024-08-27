import os
import random
import wave
import numpy as np

random_seed = 1234
random.seed(random_seed)

def get_audio_duration(file_path):
    """
    Calculate the duration of a .wav file in seconds.
    
    Parameters:
    - file_path: Path to the .wav file.
    
    Returns:
    - Duration of the file in seconds (float).
    """
    with wave.open(file_path, 'r') as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        duration = frames / float(rate)
        return duration

def create_speaker_dict(root_folder, min_duration=None):
    """
    Create a dictionary with speaker folder names as keys and a nested dictionary for left and right ear files,
    optionally filtering out any files shorter than a specified minimum duration.
    
    Parameters:
    - root_folder: Path to the root folder containing the speaker folders.
    - min_duration: Minimum duration in seconds to include a file in the dictionary (default is None, which includes all lengths).
    
    Returns:
    - A dictionary with speaker names as keys and nested dictionaries with lists of .wav file paths for each ear.
    """
    speaker_dict = {}
    
    for item in os.listdir(root_folder):
        item_path = os.path.join(root_folder, item)
        if os.path.isdir(item_path):
            ear_dict = {}
            for file in os.listdir(item_path):
                if file.endswith('.wav'):
                    file_path = os.path.join(item_path, file)
                    if min_duration is None or get_audio_duration(file_path) >= min_duration:
                        ear_indicator = file.split('_')[-2]
                        if ear_indicator not in ear_dict:
                            ear_dict[ear_indicator] = []
                        ear_dict[ear_indicator].append(file_path)
            # Assign to 'left' and 'right' based on unique values
            ear_keys = list(ear_dict.keys())
            if len(ear_keys) == 2:
                speaker_dict[item] = {'left': ear_dict[ear_keys[0]], 'right': ear_dict[ear_keys[1]]}
            else:
                print(f"Warning: Speaker {item} does not have exactly 2 unique ear indicators.")
    
    return speaker_dict

data_folder = "/mnt/009-Audio/Internships/AccelAuthentification/chinese_accel_micro_databases/separateAug/amb_0/mic"
min_audio_length = 4 # Set to None if you want to accept all lengths   
print(f'Creating pairs with min length of {min_audio_length} from {data_folder}')

data_dict = create_speaker_dict(data_folder, min_audio_length)
print("Found Speakers: ", list(data_dict.keys()))

# Creating random index pairs

positive_samples = {}
negative_samples = {}

# half of it
num_pos_examples_per_speaker = 75
num_neg_examples_per_speaker = 3

# Creating random positive pairs
for n, speaker in enumerate(list(data_dict.keys())):
    for ear in ['left', 'right']:
        if len(data_dict[speaker][ear]) > 1:
            num_of_examples = num_pos_examples_per_speaker
            all_possible_pairs = [(i, j) for i in range(len(data_dict[speaker][ear])) for j in range(len(data_dict[speaker][ear])) if i != j]
            if len(all_possible_pairs) >= num_of_examples:
                unique_pairs = random.sample(all_possible_pairs, num_of_examples)
                positive_samples[(n, ear)] = unique_pairs

# Negative pairs (code remains the same)
for i, speaker in enumerate(list(data_dict.keys())):
    if len(data_dict[speaker]['left']) > 0 or len(data_dict[speaker]['right']) > 0:
        num_of_speakers = len(list(data_dict.keys()))
        other_speakers = list(set(range(num_of_speakers)) - {i})
        num_per_speaker = num_neg_examples_per_speaker
        max_speakers = len(other_speakers)
        
        for spkr in np.random.permutation(other_speakers)[:max_speakers]:
            spkr_key = list(data_dict.keys())[spkr]
            if len(data_dict[spkr_key]['left']) > 0 or len(data_dict[spkr_key]['right']) > 0:
                for ear in ['left', 'right']:
                    if len(data_dict[speaker][ear]) > 0 and len(data_dict[spkr_key][ear]) > 0:
                        all_possible_pairs = [(i, j) for i in range(len(data_dict[speaker][ear])) for j in range(len(data_dict[spkr_key][ear]))]
                        if len(all_possible_pairs) >= num_per_speaker:
                            unique_pairs = random.sample(all_possible_pairs, num_per_speaker)
                            negative_samples[(i, spkr, ear)] = unique_pairs

# Print out the total counts of samples generated
print(f"Positive samples: ({len(positive_samples)} Speakers with Pairs)")
print(f"Negative samples: ({len(negative_samples)} Speaker Pairs)")

file_name = 'SeparateEars_amb0_4s+_' + str(random_seed) + '.txt'
# Open a text file for writing
with open(file_name, 'w') as file:
    # Positive samples
    for key in list(positive_samples.keys()):
        speaker, ear = key
        for n1, n2 in positive_samples[key]:
            speaker_name = list(data_dict.keys())[speaker]
            p1, p2 = data_dict[speaker_name][ear][n1], data_dict[speaker_name][ear][n2]

            # Write to file with 1 for positive samples, only filenames
            file.write(f"1 {os.path.abspath(p1)} {os.path.abspath(p2)}\n")

    # Negative samples
    for key in negative_samples.keys():
        for i, j in negative_samples[key]:
            sp1, sp2, ear = key
            sp1, sp2 = list(data_dict.keys())[sp1], list(data_dict.keys())[sp2]
            p1, p2 = data_dict[sp1][ear][i], data_dict[sp2][ear][j]

            # Write to file with 0 for negative samples, only filenames
            file.write(f"0 {os.path.abspath(p1)} {os.path.abspath(p2)}\n")

print("Verification Pairs Successfully Created")
