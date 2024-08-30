#
# Copyright (c) [2024] TDK U.S.A. Corporation
#
import os
import csv
import librosa
import numpy as np
from sklearn.model_selection import train_test_split

# Constants
SAMPLERATE = 16000
SEGMENT_DURATION = 3.0  # Target segment duration in seconds
AMP_THRESHOLD = 5e-04  # Amplitude threshold to filter quiet segments

def get_accel_path(original_path):
    parts = original_path.split('/')
    # Change the directory from 'mic' to 'acc'
    parts = [part if part != 'mic' else 'acc' for part in parts]
    # Add 'ACCEL_' prefix to the file name
    parts[-1] = 'ACCEL_' + parts[-1]
    return '/'.join(parts)

def segment_utterance_and_prepare_train_dev_csv(data_folder, save_folder, test_size=0.1, amp_th=AMP_THRESHOLD):
    print("Segmenting the Audio files... (might take a while depending on the size of the database)")
    segments = []
    
    # Collect segments with amplitude filtering
    for n, speaker_id in enumerate(os.listdir(data_folder)):
        # if n>3:
        #     break
        
        # progress
        if n%20 == 0:
            print(f"{n} speakers have been processed so far")
        speaker_path = os.path.join(data_folder, speaker_id)
        if not os.path.isdir(speaker_path):
            continue
            
        for utterance_file in os.listdir(speaker_path):
            utterance_path = os.path.join(speaker_path, utterance_file)
            if not utterance_path.endswith('.wav'):
                continue
            
            full_utterance_path = os.path.abspath(utterance_path).replace("\\", "/")
            accel_utterance_path = get_accel_path(full_utterance_path)
            # print(accel_utterance_path)
            # print(accel_utterance_path)
            
            audio, sr = librosa.load(full_utterance_path, sr=SAMPLERATE)
            total_samples = audio.shape[0]
            audio_duration = librosa.get_duration(y=audio, sr=sr)
            target_samples = int(sr * SEGMENT_DURATION)


            num_segments = total_samples // target_samples
            possible_seg = total_samples / target_samples

            # a threshold for applyinh the final window if we have a considerable amount of audio left
            if num_segments >= 1 and abs(possible_seg - num_segments) >= 0.5:
                num_segments+=1
               
                
            for segment_idx in range(num_segments):
                start_sample = segment_idx * target_samples
                stop_sample = start_sample + target_samples
                segment = audio[start_sample:stop_sample]

                if segment_idx == num_segments-1:
                    start_sample = total_samples - target_samples
                    stop_sample = total_samples
                    segment = audio[-target_samples:]
                
                # Check if the segment's average amplitude is above the threshold
                if np.mean(np.abs(segment)) >= amp_th:
                    segment_id = f"{speaker_id}--{utterance_file}--{segment_idx}_0_{len(segment)*1000.0}"
                    segments.append([segment_id, audio_duration, (full_utterance_path, accel_utterance_path), start_sample, stop_sample, speaker_id])
    
    # Split segments into training and development sets
    train_segments, dev_segments = train_test_split(segments, test_size=test_size, random_state=42)
    
    # Save to CSV files
    train_csv_path = os.path.join(save_folder, "fused_train.csv")
    dev_csv_path = os.path.join(save_folder, "fused_dev.csv")
    print("Writing the csv files...")
    for csv_path, data in zip([train_csv_path, dev_csv_path], [train_segments, dev_segments]):
        with open(csv_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["ID", "duration", "wav", "start", "stop", "spk_id"])
            for row in data:
                csvwriter.writerow(row)

# Example usage:
data_folder = "/mnt/009-Audio/Internships/AccelAuthentification/chinese_accel_micro_databases/Farfield_aug/mic"
save_folder = "/mnt/009-Audio/Internships/AccelAuthentification/chinese_accel_micro_databases/Farfield_aug/mic"
segment_utterance_and_prepare_train_dev_csv(data_folder, save_folder, test_size=0.2)
