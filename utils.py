import torch
import torchaudio
import os
from speechbrain.processing.features import DynamicRangeCompression, InputNormalization
from speechbrain.lobes.features import Fbank, MFCC, DCT
from speechbrain.utils.metric_stats import minDCF
import speechbrain.utils.metric_stats as sbm
from sklearn.metrics import roc_curve
import numpy as np
import matplotlib.pyplot as plt
from speechbrain.lobes.features import Fbank
from scipy.signal import butter, filtfilt
from datetime import datetime


def add_white_gaussian_noise(waveform, desired_snr):

    # Calculate the power of the signal
    signal_power = waveform.pow(2).mean()
    
    # Calculate the required noise power for desired SNR
    snr_linear = 10 ** (desired_snr / 10)
    noise_power = signal_power / snr_linear
    
    # Generate white Gaussian noise
    noise = torch.randn(waveform.size()) * torch.sqrt(noise_power)
    
    # Add the noise to the original signal
    noisy_signal = waveform + noise
    
    return noisy_signal

# Usage example
# noisy_waveform, sr = add_white_gaussian_noise('path_to_audio.wav', 10)

def audio_preprocess(audio, accel_data = False):
    HPb, HPa = butter(2, 35/(16000/2), btype='high', analog = False)
    # accel preprocess
    if accel_data:
        filtered_audio = filtfilt(HPb, HPa, audio.numpy())
        filtered_audio *= (0.8070*3.16*2)
    
    # mic preprocess
    else:
        filtered_audio = filtfilt(HPb, HPa, audio.numpy())
        filtered_audio *= 0.8070
    return torch.tensor(filtered_audio.copy(), dtype = torch.float32)



def mic_lowpass(audio):
    HPb, HPa = butter(2, 2000/(16000/2), btype='low', analog = False)
    filtered_audio = filtfilt(HPb, HPa, audio.numpy())


    return torch.tensor(filtered_audio.copy(), dtype = torch.float32)

def apply_fade(audio, sr = 16000, fade_in_time=0.1, fade_out_time=0.1):
    """
    Apply a linear fade-in and fade-out to the audio tensor.
    
    Parameters:
    audio (torch.Tensor): The input audio tensor.
    sr (int): Sample rate of the audio.
    fade_in_time (float): Length of the fade-in in seconds.
    fade_out_time (float): Length of the fade-out in seconds.
    """
    fade_in_samples = int(sr * fade_in_time)
    fade_out_samples = int(sr * fade_out_time)

    # Create fade-in envelope
    fade_in = torch.linspace(0, 1, fade_in_samples).to(audio.device)

    # Create fade-out envelope
    fade_out = torch.linspace(1, 0, fade_out_samples).to(audio.device)

    # Apply fade-in
    audio[:fade_in_samples] *= fade_in

    # Apply fade-out
    audio[-fade_out_samples:] *= fade_out

    return audio.unsqueeze(0)

def plot_frr_far_vs_threshold(labels, scores, file_path=''):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    frr = 1 - tpr  # False Rejection Rate is 1 - True Positive Rate

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, frr, label='FRR (1-TPR)', color='red')
    plt.plot(thresholds, fpr, label='FAR (FPR)', color='blue')
    plt.xlabel('Decision Threshold')
    plt.ylabel('Error Rate')
    # plt.title('FRR and FAR vs. Threshold')
    plt.legend(loc="upper right")
    plt.grid(True)

    # Find the EER point
    eer_index = np.nanargmin(np.abs(frr - fpr))
    eer_threshold = thresholds[eer_index]
    eer_value = fpr[eer_index]
    plt.scatter(eer_threshold, eer_value, color='green')
    plt.annotate(f'EER = {eer_value:.2f}', (eer_threshold, eer_value), textcoords="offset points", xytext=(0,10), ha='center')

    output_path = os.path.join(file_path, 'frr_far_plot.png')
    plt.savefig(output_path)
    plt.close()

    return eer_value, eer_threshold


def plot_roc_curve(fpr, tpr, label=None, file_path = ''):
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')  # Dashed diagonal

    # Calculate the EER and annotate it
    eer_index = np.nanargmin(np.abs(fpr - (1 - tpr)))
    eer = fpr[eer_index]
    plt.scatter(fpr[eer_index], tpr[eer_index], color='red')  # EER point
    plt.annotate(f'EER = {eer:.2f}', (fpr[eer_index], tpr[eer_index]), textcoords="offset points", xytext=(10,-10), ha='center')

    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve with EER Point')
    plt.legend(loc="lower right")
    plt.grid(True)

    output_path = os.path.join(file_path, 'roc_curve.png')
    plt.savefig(output_path)
    plt.close()

def plot_det_curve(fpr, fnr, label=None, file_path = ''):
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, fnr, linewidth=2, label=label)
    plt.xlabel('False Positive Rate')
    plt.ylabel('False Negative Rate')
    plt.title('DET Curve')
    # plt.xscale('log')
    # plt.yscale('log')
    plt.grid(True)

    output_path = os.path.join(file_path, 'det_curve.png')
    plt.savefig(output_path)
    plt.close()


def calculate_eer_and_curves(labels, scores, root):
    fpr, tpr, threshold = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    
    plot_roc_curve(fpr, tpr, "ROC Curve", root)
    plot_det_curve(fpr, fnr, "DET Curve", root)
    _, _ = plot_frr_far_vs_threshold(labels, scores, root)


    return EER, eer_threshold

def calculate_minDCF(labels, scores):
    # Considering that labels start with 1's and end with 0 s (not ramdom)
    last_pos_index = sum(labels)
    pos_scores = scores[:last_pos_index]
    neg_scores = scores[last_pos_index:]

    # print('Positive pairs:', len(pos_scores), 'Last_pos_index =', last_pos_index)
    # print('SpeechBrain EER = ', sbm.EER(torch.tensor(pos_scores), torch.tensor(neg_scores) ))

    return minDCF(torch.tensor(pos_scores), torch.tensor(neg_scores))


def extract_features(wv):
    n_mel_channel= 80
    hop_length= 256
    win_length= 1024
    n_fft= 1024
    mel_fmin= 0.0
    mel_fmax= 8000.0
    mel_normalized= False
    power= 1
    norm="slaney"
    mel_scale= "slaney"
    drc = DynamicRangeCompression()
    IpNorm = InputNormalization(norm_type="sentence", std_norm=False)

    with torch.no_grad():
        mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft = n_fft, win_length= win_length, hop_length=hop_length, f_min=mel_fmin, f_max=mel_fmax,
                                                    n_mels=n_mel_channel, power = power, normalized=mel_normalized, norm = norm, mel_scale=mel_scale)
        ft2 =  drc(mel_spec(wv))
        inp_len = torch.ones([ft2.size(0)])
        features = IpNorm(ft2, inp_len)
    
    # mel_fbank = Fbank(n_mels=80)
    # features = mel_fbank(wv).transpose(1, 2)
    return features



def extract_feature_Fbank(wv, is_MFCC = True, n_MFCC = 13, n_mel = 80):
        
        # drc = DynamicRangeCompression()
        IpNorm = InputNormalization(norm_type="sentence", std_norm=False)
        fbank = Fbank(n_mels= n_mel)
        
        with torch.no_grad():
            ft = fbank(wv)

            if is_MFCC:
                compute_dct = DCT(ft.size(-1), n_out = n_MFCC)
                ft = compute_dct(ft)

            
            inp_len = torch.ones([ft.size(0)])

            # checked its not doing anything if single
            features = IpNorm(ft, inp_len)

        return features
            
def compute_embedding(feat, embedding_model, device):

    embeddings = embedding_model(feat.transpose(1, 2).to(device))

    return(embeddings)

def compute_cosine_similarity(emb1, emb2):

    cosine_func =  torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    return (cosine_func(emb1, emb2).item())


def read_verif_file(file_path, abs_path = True, base_path = None):
    paths, label = [], []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip().split()

            class_label = int(line[0])
            audio_path1 = line[1]
            audio_path2 = line[2]

            if abs_path is False:
                audio_path1 = os.path.join(base_path, audio_path1)
                audio_path2 = os.path.join(base_path, audio_path2)


            paths.append((audio_path1, audio_path2))
            label.append(class_label)
    
    return paths, label


def calculate_eer(labels, scores):
    fpr, tpr, threshold = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

    return EER, eer_threshold


def feats_fusion(feat1, feat2, batch = False):

    if batch:
        fused = torch.concat((feat1, feat2), dim = 2)
    else:
        fused = torch.concat((feat1, feat2), dim = 1)
    return fused


def read_audio(p, normalize = True):
    return torchaudio.load(p, normalize= normalize)[0][0, :].unsqueeze(0)


def save_pairs_scores(paths, labels, scores, root, missed = False):
    output_path = os.path.join(root, "all_pairs.txt")
    if missed:
        output_path = os.path.join(root, "missclassified_pairs.txt")
    with open(output_path, 'w') as file:
        for j, (lab, score) in enumerate(zip(labels, scores)):
                file.write(f"{lab} {score}, {paths[j]}\n")


def extract_missclassified(scores, labels, thresh, pths):
    missed_paths, missed_labels, missed_scores = [], [], []

    for j, (lab, score) in enumerate(zip(labels, scores)):
        if (lab == 1 and score < thresh) or (lab == 0 and score > thresh):
            missed_paths.append(pths[j])
            missed_labels.append(lab)
            missed_scores.append(score)
    
    return missed_paths, missed_labels, missed_scores


def save_log(mic_p, acc_p, model_p, feature_mode, EER, eer_thresh, min_dcf, dcf_thresh, root, pairs_len):
    output_path = os.path.join(root, "verification_log.txt")
    with open(output_path, 'w') as file:
        file.write(datetime.now().strftime('%H:%M %d/%m/%Y\n'))
        file.write(f'model_path: {model_p}\nmic_path: {mic_p}\nacc_path: {acc_p}\ninput_feature_mode: {feature_mode}\n') 
        file.write(f'Number of pairs = {pairs_len}\nEER = {EER}, EER Threshold = {eer_thresh}\nmin_DCF = {min_dcf}, min_DCF Threshold = {dcf_thresh}')
