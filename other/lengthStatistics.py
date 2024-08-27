import os
import torchaudio
import pandas as pd
import matplotlib.pyplot as plt

# Function to find all .wav files
def find_wav_files(root_folder):
    wav_files = []
    for subdir, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".wav"):
                wav_files.append(os.path.join(subdir, file))
    return wav_files

# Function to get duration of a single .wav file
def get_duration(wav_path):
    waveform, sample_rate = torchaudio.load(wav_path)
    return waveform.shape[1] / sample_rate  # Duration in seconds

# Analyzing audio files in a given directory
def analyze_audio_files(folder_path):
    files = find_wav_files(folder_path)
    durations = []
    
    for file in files:
        duration = get_duration(file)
        durations.append(duration)
    
    return durations

# Plotting histograms for two directories
def plot_histograms(durations1, durations2, labels):
    plt.figure(figsize=(10, 6))
    plt.hist(durations1, bins=30, color='blue', alpha=0.5, label=labels[0])
    # plt.hist(durations2, bins=30, color='red', alpha=0.5, label=labels[1])
    # plt.title('Comparison of Audio File Durations')
    plt.xlabel('Duration in Seconds')
    plt.ylabel('Frequency')
    # plt.legend(loc='upper right')
    # plt.grid(True)
    plt.savefig('combined_histogram.png')
    plt.close()

# Paths to the two directories
folder_path1 = '/mnt/009-Audio/Internships/AccelAuthentification/chinese_accel_micro_databases/speaker_splits/split1/mic/train'
# folder_path2 = '/mnt/009-Audio/Internships/AccelAuthentification/chinese_accel_micro_databases/extracted_segments/Mic_test'

# Analyze both directories
durations1 = analyze_audio_files(folder_path1)
# durations2 = analyze_audio_files(folder_path2)

# Plot histograms together
plot_histograms(durations1, None, ['VOX', 'Ours'])

