#
# Copyright (c) [2024] TDK U.S.A. Corporation
#
import os
import numpy as np
import shutil
import soundfile as sf
from scipy.signal import fftconvolve, convolve

rir_paths = "/mnt/009-Audio/Internships/AccelAuthentification/RIRS_NOISES/acc_noises"
input_dir_path = "/mnt/009-Audio/Internships/AccelAuthentification/chinese_accel_micro_databases/speaker_splits/split1/accel/test"
speakers_dict = {}

    
def matchSize(wav_len, noise):
    if len(noise) >= wav_len:
        output = noise[:wav_len]
    else:
        num_of_copies  = 1 + (wav_len // len(noise))
        output = np.tile(noise, num_of_copies)

        end_ind = wav_len
        output = output[:end_ind]
        
    if len(output) != wav_len:
        print("Checking outputs", len(output), wav_len)


    
    return output



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




print("Loading the noise samples...")
noise_set = load_rir_dataset(rir_paths)
noise_len = len(noise_set) 

print(f'Found {noise_len} samples')


for item in os.listdir(input_dir_path):
    item_path = os.path.join(input_dir_path, item)

    if os.path.isdir(item_path):
    
        for file in os.listdir(item_path):
            if file.endswith(".wav"):

                # speakers_dict[item].append(os.path.join(item_path, file))
                original_path = os.path.join(item_path, file)
                new_path = original_path.replace('/accel/', '/spoofed_acc/')

                # Making the directory
                os.makedirs(item_path.replace('/accel/', '/spoofed_acc/'), exist_ok=True)

                # Picking up a random RIR
                chosen_noise = noise_set[np.random.randint(noise_len)]
                original_audio, sr = sf.read(original_path)

                spoofed_audio = matchSize(len(original_audio), chosen_noise)


                sf.write(new_path, spoofed_audio, sr)

# print(speakers_dict)
# for item in os.listdir(input_dir_path):
    # print(item)
    
# print(speakers_dict)
 
