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

data_folder = "/mnt/009-Audio/Internships/AccelAuthentification/chinese_accel_micro_databases/extracted_segments/Mic_test"
data_dict = create_speaker_dict(data_folder)

chosen_speaker = "0346"  # Replace "SpeakerName" with your chosen speaker's folder name

positive_samples = []
negative_samples = []

# Check if the chosen speaker exists in the dictionary
if chosen_speaker in data_dict:
    # Create positive pairs (from the same speaker)
    for i in range(100):  # You can adjust the number of pairs
        pair = random.sample(data_dict[chosen_speaker], 2)
        positive_samples.append((pair[0], pair[1]))

    # Create negative pairs (chosen speaker with other speakers)
    other_speakers = [sp for sp in data_dict if sp != chosen_speaker]
    for other_speaker in other_speakers:
        for i in range(50):  # Number of negative samples per other speaker
            file1 = random.choice(data_dict[chosen_speaker])
            file2 = random.choice(data_dict[other_speaker])
            negative_samples.append((file1, file2))

# Print samples to console or write them to a file
print("Positive Samples:", positive_samples)
print("Negative Samples:", negative_samples)

# Writing samples to a file
with open('verification_pairs.txt', 'w') as file:
    for p1, p2 in positive_samples:
        file.write(f"1 {os.path.abspath(p1)} {os.path.abspath(p2)}\n")
    for p1, p2 in negative_samples:
        file.write(f"0 {os.path.abspath(p1)} {os.path.abspath(p2)}\n")

print("Verification Pairs Successfully Created")
