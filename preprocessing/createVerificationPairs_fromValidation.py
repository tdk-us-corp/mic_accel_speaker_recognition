import csv
import random
from itertools import combinations

def generate_pairs(csv_file_path, output_file_path, num_pairs=20000):
    # Dictionary to hold data by speaker ID
    speaker_data = {}

    # Reading the CSV file
    with open(csv_file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)

        next(reader)
        for row in reader:
            duration = float(row[1])
            if duration >= 4.0:  # Only consider audio samples that are 4 seconds or longer
                spk_id = row[5]
                wav_info = eval(row[2])
                if spk_id in speaker_data:
                    speaker_data[spk_id].append(wav_info[0])
                else:
                    speaker_data[spk_id] = [wav_info[0]]

    # Generate positive pairs
    positive_pairs = []
    for spk_id, wavs in speaker_data.items():
        if len(wavs) > 1:
            # Generate all possible pairs from the speaker's recordings
            for pair in combinations(wavs, 2):
                positive_pairs.append(pair)
                if len(positive_pairs) >= num_pairs:
                    break
        if len(positive_pairs) >= num_pairs:
            break
    random.shuffle(positive_pairs)
    positive_pairs = positive_pairs[:num_pairs]

    # Generate negative pairs
    negative_pairs = []
    speaker_ids = list(speaker_data.keys())
    while len(negative_pairs) < num_pairs:
        spk_id1, spk_id2 = random.sample(speaker_ids, 2)
        if spk_id1 != spk_id2:
            wav1 = random.choice(speaker_data[spk_id1])
            wav2 = random.choice(speaker_data[spk_id2])
            negative_pairs.append((wav1, wav2))

    # Writing to output file
    with open(output_file_path, 'w') as f:
        for pair in positive_pairs:
            f.write(f'1 {pair[0]} {pair[1]}\n')
        for pair in negative_pairs:
            f.write(f'0 {pair[0]} {pair[1]}\n')

# Usage

generate_pairs('fused_dev_fixed.csv', 'validation_pairs.txt')
