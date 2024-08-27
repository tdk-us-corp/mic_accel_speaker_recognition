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
    Create a dictionary with speaker folder names as keys and a list of .wav file paths as values,
    optionally filtering out any files shorter than a specified minimum duration.
    
    Parameters:
    - root_folder: Path to the root folder containing the speaker folders.
    - min_duration: Minimum duration in seconds to include a file in the dictionary (default is None, which includes all lengths).
    
    Returns:
    - A dictionary with speaker names as keys and lists of .wav file paths as values.
    """
    speaker_dict = {}
    
    for item in os.listdir(root_folder):
        item_path = os.path.join(root_folder, item)
        if os.path.isdir(item_path):
            speaker_dict[item] = []
            for file in os.listdir(item_path):
                if file.endswith('.wav'):
                    file_path = os.path.join(item_path, file)
                    if min_duration is None or get_audio_duration(file_path) >= min_duration:
                        speaker_dict[item].append(file_path)
    
    return speaker_dict

data_folder = "/mnt/009-Audio/Internships/AccelAuthentification/chinese_accel_micro_databases/speaker_splits/split1/mic/test"
min_audio_length = 4 # Set to None if you want to accept all lengths   

data_dict = create_speaker_dict(data_folder, min_audio_length)
print("Found Speakers: ", list(data_dict.keys()))

# Creating random index pairs

positive_samples = {}
negative_samples = {}

num_pos_examples_per_speaker= 50
num_neg_examples_per_speaker = 5
# creating random positive pairs
for n, speaker in enumerate(list(data_dict.keys())):
    if len(data_dict[speaker]) > 1:
        num_of_examples = num_pos_examples_per_speaker
        all_possible_pairs = [(i, j) for i in range(len(data_dict[speaker])) for j in range(len(data_dict[speaker])) if i != j]
        if len(all_possible_pairs) >= num_of_examples:
            unique_pairs = random.sample(all_possible_pairs, num_of_examples)
            positive_samples[n] = unique_pairs

# negative pairs
for i, speaker in enumerate(list(data_dict.keys())):
    if len(data_dict[speaker]) > 1:
        num_of_speakers = len(list(data_dict.keys())) 
        other_speakers = list(set(range(num_of_speakers)) - {i})
        num_per_speaker = num_neg_examples_per_speaker
        max_speakers = len(other_speakers)
        # max_speakers = 40
        # print(np.random.permutation(other_speakers))

        for spkr in np.random.permutation(other_speakers)[:max_speakers]:
            spkr_key = list(data_dict.keys())[spkr]
            if len(data_dict[spkr_key]) > 0:
                all_possible_pairs = [(i, j) for i in range(len(data_dict[speaker])) for j in range(len(data_dict[spkr_key]))]
                if len(all_possible_pairs) >= num_per_speaker:
                    unique_pairs = random.sample(all_possible_pairs, num_per_speaker)
                    negative_samples[(i, spkr)] = unique_pairs

# Print out the total counts of samples generated
print(f"Positive samples: ({len(positive_samples)} Speakers with Pairs)")
print(f"Negative samples: ({len(negative_samples)} Speaker Pairs)")

file_name = 'mus_0_4+_' + str(random_seed) + '.txt'
# Open a text file for writing
with open(file_name, 'w') as file:
    # Positive samples
    for key in list(positive_samples.keys()):
        for n1, n2 in positive_samples[key]:
            speaker = list(data_dict.keys())[key]

            p1, p2 = data_dict[speaker][n1], data_dict[speaker][n2]

            # Write to file with 1 for positive samples, only filenames
            # print(f"1 {os.path.abspath(p1)} {os.path.abspath(p2)}\n")
            file.write(f"1 {os.path.abspath(p1)} {os.path.abspath(p2)}\n")

    # Negative samples
    for key in negative_samples.keys():
        for i, j in negative_samples[key]:
            sp1, sp2 = key
            sp1, sp2 = list(data_dict.keys())[sp1], list(data_dict.keys())[sp2]
            p1, p2 = data_dict[sp1][i], data_dict[sp2][j]

            # Write to file with 0 for negative samples, only filenames
            # print(f"0 {os.path.abspath(p1)} {os.path.abspath(p2)}\n")
            file.write(f"0 {os.path.abspath(p1)} {os.path.abspath(p2)}\n")

print("Verification Pairs Successfully Created")
