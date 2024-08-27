import os
import sys
import torch
import logging
import torchaudio
import speechbrain as sb
from tqdm.contrib import tqdm
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.data_utils import download_file
from speechbrain.utils.distributed import run_on_main
from speechbrain.processing.features import DynamicRangeCompression
from speechbrain.processing.features import InputNormalization
from speechbrain.lobes.features import Fbank, MFCC, DCT

from sklearn.metrics import roc_curve
import numpy as np
import matplotlib.pyplot as plt
from speechbrain.lobes.features import Fbank
from scipy.signal import butter, filtfilt

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

def plot_frr_far_vs_threshold(labels, scores, file_path='frr_far_plot.png'):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    frr = 1 - tpr  # False Rejection Rate is 1 - True Positive Rate

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, frr, label='FRR (1-TPR)', color='red')
    plt.plot(thresholds, fpr, label='FAR (FPR)', color='blue')
    plt.xlabel('Decision Threshold')
    plt.ylabel('Error Rate')
    plt.title('FRR and FAR vs. Threshold')
    plt.legend(loc="upper right")
    plt.grid(True)

    # Find the EER point
    eer_index = np.nanargmin(np.abs(frr - fpr))
    eer_threshold = thresholds[eer_index]
    eer_value = fpr[eer_index]
    plt.scatter(eer_threshold, eer_value, color='green')
    plt.annotate(f'EER = {eer_value:.2f}', (eer_threshold, eer_value), textcoords="offset points", xytext=(0,10), ha='center')

    plt.savefig(file_path)
    plt.close()

    return eer_value, eer_threshold


def plot_roc_curve(fpr, tpr, label=None, file_path='roc_curve.png'):
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
    plt.savefig(file_path)
    plt.close()

def plot_det_curve(fpr, fnr, label=None, file_path='det_curve.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, fnr, linewidth=2, label=label)
    plt.xlabel('False Positive Rate')
    plt.ylabel('False Negative Rate')
    plt.title('DET Curve')
    # plt.xscale('log')
    # plt.yscale('log')
    plt.grid(True)
    plt.savefig(file_path)
    plt.close()


def calculate_eer_and_curves(labels, scores):
    fpr, tpr, threshold = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    
    plot_roc_curve(fpr, tpr, "ROC Curve", "roc_curve.png")
    plot_det_curve(fpr, fnr, "DET Curve", "det_curve.png")
    _, _ = plot_frr_far_vs_threshold(labels, scores)


    return EER, eer_threshold


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
            
def compute_embedding(feat):

    embeddings = params["embedding_model"](feat.transpose(1, 2).to(run_opts["device"]))

    return(embeddings)

def compute_cosine_similarity(emb1, emb2):
    cosine_func =  torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    return (cosine_func(emb1, emb2).item())


def read_verif_file(file_path):
    paths, label = [], []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip().split()
            class_label = int(line[0])
            audio_path1 = line[1]
            audio_path2 = line[2]

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

    

# Logger setup
logger = logging.getLogger(__name__)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

# Load hyperparameters file with command-line overrides
params_file, run_opts, overrides = sb.core.parse_arguments(sys.argv[1:])
with open(params_file) as fin:
    params = load_hyperpyyaml(fin, overrides)

# Download verification list (to exclude verification sentences from train)
# veri_file_path = os.path.join(
#     params["save_folder"], os.path.basename(params["verification_file"])
# )

# Create experiment directory
sb.core.create_experiment_directory(
    experiment_directory=params["output_folder"],
    hyperparams_to_save=params_file,
    overrides=overrides,
)

# run_on_main(params["pretrainer"].collect_files)

params["pretrainer"].load_collected()
params["embedding_model"].eval()
params["embedding_model"].to(run_opts["device"])
print(type(params["embedding_model"]))

# Computing  enrollment and test embeddings
logger.info("Computing user embeddings...")


dir_path = "/mnt/009-Audio/Internships/AccelAuthentification/chinese_accel_micro_databases/speaker_splits/split1/mic/test"





all_scores = []
sanity_check = True
preprocess = True
fading = False
hstack = False
feature_mask = False
delta = False

hstack_frames_added = 30
# for embedding norm if necessary
embNorm = InputNormalization(norm_type="global", std_norm=False)

# feat_mode = "Fbank"
feat_mode = "Fbank"
audio_norm = True
print(f"feature extraction mode is {feat_mode}")


mic_MFCC = 13
acc_MFCC = 13


corrupted_counter = 0
audio_per_speaker = 50


## Modality Config
mic_only = False
acc_only = True
fused = False

print(f'Input modality config:\n mic: {mic_only} \n acc: {acc_only} \n fused: {fused}')



for dir in os.listdir(dir_path):
    folder_path = os.path.join(dir_path, dir)
    acc_folder_path = os.path.join(dir_path, dir).replace('/mic/', '/accel/')

    if os.path.isdir(folder_path):

        embedding_dim = 192
        embedding_vectors = torch.empty((audio_per_speaker, embedding_dim), dtype = torch.float32, device = "cpu")

        print(f'Extracting the embeddings from user {dir}')
        added_audios = 0 

        for _, file in enumerate(os.listdir(folder_path)):
            

            if added_audios < audio_per_speaker:
                if file.endswith('.wav'):


                    # Files have to fully aligned

                    file_path = os.path.join(folder_path, file)
                    acc_file_path= os.path.join(acc_folder_path, 'ACCEL_', file).replace('ACCEL_/', 'ACCEL_')
                    
                    # reading the audio files
                    mic_audio_p1, sr = torchaudio.load(file_path, normalize= audio_norm)

                    duration_seconds = mic_audio_p1.shape[1] / sr

                    if duration_seconds < 2:
                        continue

                    mic_audio_p1 = mic_audio_p1[0, :].unsqueeze(0)
                        
                    if preprocess:
                        #  Mic preprocess
                        mic_audio_p1 = audio_preprocess(mic_audio_p1, accel_data= False)
                    
                    # Feature Extraction
                    mic_p1_feats = extract_feature_Fbank(mic_audio_p1, n_MFCC = mic_MFCC).transpose(1, 2)



                    if fused:
                            acc_audio_p1 = torchaudio.load(acc_file_path, normalize= audio_norm)[0][0, :].unsqueeze(0)
                            if preprocess:
                                acc_audio_p1 = audio_preprocess(acc_audio_p1, accel_data = True)
                            acc_p1_feats = extract_feature_Fbank(acc_audio_p1, n_MFCC = acc_MFCC).transpose(1, 2)
                            fused_feats_p1 = feats_fusion(mic_p1_feats, acc_p1_feats, batch = False)
                            p1_emb = compute_embedding(fused_feats_p1)
                        

                    if mic_only:
                        # Just mic
                        p1_emb = compute_embedding(mic_p1_feats)

                    if acc_only:
                            
                        acc_audio_p1 = torchaudio.load(acc_file_path, normalize= audio_norm)[0][0, :].unsqueeze(0)
                        if preprocess:
                            acc_audio_p1 = audio_preprocess(acc_audio_p1, accel_data = True)
                        acc_p1_feats = extract_feature_Fbank(acc_audio_p1, n_MFCC = acc_MFCC).transpose(1, 2)
                        p1_emb = compute_embedding(acc_p1_feats)

                    print(acc_file_path)



                    embedding_vectors[added_audios] = p1_emb.detach().to("cpu")
                    added_audios += 1

        new_path = folder_path.replace('/mic/', '/embeddings/')

        # IMMPORTANT PAY ATTENTION TO WHAT YOU ARE CHANGING MIGHT REWRITE
        os.makedirs(new_path, exist_ok=True)




        if fused:
            torch.save(torch.mean(embedding_vectors, dim = 0), os.path.join(new_path, f'{dir}_averaged_embedding_fused.pt'))
        if mic_only:
            torch.save(torch.mean(embedding_vectors, dim = 0), os.path.join(new_path, f'{dir}_averaged_embedding_mic.pt'))
        if acc_only:
            torch.save(torch.mean(embedding_vectors, dim = 0), os.path.join(new_path, f'{dir}_averaged_embedding_acc.pt'))



