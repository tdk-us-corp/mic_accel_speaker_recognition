import os
import numpy as np
import shutil
import soundfile as sf
from scipy.signal import fftconvolve, convolve

rir_paths = "/mnt/009-Audio/Internships/AccelAuthentification/RIRS_NOISES/simulated_rirs/mediumroom"
input_dir_path = "/mnt/009-Audio/Internships/AccelAuthentification/chinese_accel_micro_databases/speaker_splits/split1/mic/test"
speakers_dict = {}

    
def spoofAudio(input_wav, rir):
    spoofed_audio = convolve(input_wav, rir, mode = "full")

    return spoofed_audio[:input_wav.shape[0]]
    
    
def load_rir_dataset(dataset_path):
    rirs = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".wav"):
                rir_path = os.path.join(root, file)
                # print(rir_path)
                rir, _ = sf.read(rir_path)
                rirs.append(rir)
    return rirs




print("Loading the rir dataset...")
rir_set = load_rir_dataset(rir_paths)
# rir_set = rir_set[3:100]
rirs_len = len(rir_set) 

print(f'Found {rirs_len} rirs')


for item in os.listdir(input_dir_path):
    item_path = os.path.join(input_dir_path, item)

    if os.path.isdir(item_path):
    
        for file in os.listdir(item_path):
            if file.endswith(".wav"):

                # speakers_dict[item].append(os.path.join(item_path, file))
                original_path = os.path.join(item_path, file)
                new_path = original_path.replace('/mic/', '/spoofed_mic_mediumroom/')

                # Making the directory
                os.makedirs(item_path.replace('/mic/', '/spoofed_mic_mediumroom/'), exist_ok=True)

                # Picking up a random RIR
                # random_rir = rir_set[np.random.randint(rirs_len)][:, 0]
                random_rir = rir_set[np.random.randint(rirs_len)]
                original_audio, sr = sf.read(original_path)
                spoofed_audio = spoofAudio(original_audio, random_rir)

                # print(new_path)
                sf.write(new_path, spoofed_audio, sr)

# print(speakers_dict)
# for item in os.listdir(input_dir_path):
    # print(item)
    
# print(speakers_dict)
 
