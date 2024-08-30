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
SEGMENT_DURATION = 2.0  # Target segment duration in seconds
AMP_THRESHOLD = 5e-04  # Amplitude threshold to filter quiet segments



def segment_utterance_and_prepare_train_dev_csv(data_folder, save_folder, test_size=0.1, amp_th=AMP_THRESHOLD):
    print("Segmenting the Audio files... (might take a while depending on the size of the database)")
    segments = []
    
    # Collect segments with amplitude filtering
    for speaker_id in os.listdir(data_folder):
        speaker_path = os.path.join(data_folder, speaker_id)
        if not os.path.isdir(speaker_path):
            continue
            
        for utterance_file in os.listdir(speaker_path):
            utterance_path = os.path.join(speaker_path, utterance_file)
            if not utterance_path.endswith('.wav'):
                continue
            
            # Normalize the full path for cross-platform compatibility
            full_utterance_path = os.path.abspath(utterance_path)
            full_utterance_path = full_utterance_path.replace("\\", "/")  # Ensure forward slashes
            
            audio, sr = librosa.load(full_utterance_path, sr=SAMPLERATE)
            total_samples = audio.shape[0]
            audio_duration = librosa.get_duration(y=audio, sr=sr)

            target_samples = int(sr * SEGMENT_DURATION)
               
            # Sliding the last segment
            possible_seg = total_samples / target_samples
            num_segments = total_samples // target_samples
            
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
                    segments.append([segment_id, audio_duration, full_utterance_path, start_sample, stop_sample, speaker_id])
    
    # Split segments into training and development sets BASE WAS 42
    train_segments, dev_segments = train_test_split(segments, test_size=test_size, random_state=42)
    
    # saave to CSV files
    train_csv_path = os.path.join(save_folder, "train.csv")
    dev_csv_path = os.path.join(save_folder, "dev.csv")
    

    print("Writing the csv files...")
    for csv_path, data in zip([train_csv_path, dev_csv_path], [train_segments, dev_segments]):
    
        with open(csv_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["ID", "duration", "wav", "start", "stop", "spk_id"])
            csvwriter.writerows(data)




# Example usage:
data_folder = "/mnt/009-Audio/Internships/AccelAuthentification/chinese_accel_micro_databases/speaker_splits/split1/mic/augmented_train/train"
save_folder = "/mnt/009-Audio/Internships/AccelAuthentification/chinese_accel_micro_databases/speaker_splits/split1/mic/augmented_train/train"
segment_utterance_and_prepare_train_dev_csv(data_folder, save_folder, test_size=0.2)
