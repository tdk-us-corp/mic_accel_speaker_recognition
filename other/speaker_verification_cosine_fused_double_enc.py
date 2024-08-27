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
            
def compute_embedding(feat, emb_model_name):

    embeddings = params[emb_model_name](feat.transpose(1, 2).to(run_opts["device"]))

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
params["acc_embedding_model"].eval()
params["embedding_fuser"].eval()

params["embedding_model"].to(run_opts["device"])
params["acc_embedding_model"].to(run_opts["device"])
params["embedding_fuser"].to(run_opts["device"])

print(type(params["embedding_model"]))

# Computing  enrollment and test embeddings
logger.info("Computing enroll/test embeddings...")



mic_path = "verif_pairs/paper_pairs/SeparateEars_4s+_1234.txt"
accel_path = "verif_pairs/paper_pairs/SeparateEars_4s+_1234_accel.txt"

mic_pths, mic_labels = read_verif_file(mic_path)
acc_pths, acc_labels = read_verif_file(accel_path)

pairs_len = len(mic_labels)

print(f"using the {mic_path} and {accel_path} files with {pairs_len} pairs")


all_scores = []
sanity_check = True
preprocess = True
fading = False
hstack = False
feature_mask = False

hstack_frames_added = 30
# for embedding norm if necessary
embNorm = InputNormalization(norm_type="global", std_norm=False)

# feat_mode = "Fbank"
feat_mode = "Fbank"
audio_norm = True
print(f"feature extraction mode is {feat_mode}")


mic_MFCC = 20
acc_MFCC = 20

# Files have to fully aligned
with open('pairs_scores.txt', 'w') as file:
    for i, (p1, p2) in enumerate(mic_pths):
        # Sanity check the name of two audio files
        if sanity_check:
            if acc_pths[i][0].split("/")[-1][6:] != p1.split("/")[-1] or  acc_pths[i][1].split("/")[-1][6:] != p2.split("/")[-1]:
                print("Warning! audio files might not be correctly matched")
        acc_p1 = acc_pths[i][0]
        acc_p2 = acc_pths[i][1]

        # reading the audio files
        mic_audio_p1 = torchaudio.load(p1, normalize= audio_norm)[0][0, :].unsqueeze(0)
        acc_audio_p1 = torchaudio.load(acc_p1, normalize= audio_norm)[0][0, :].unsqueeze(0)

        mic_audio_p2 = torchaudio.load(p2, normalize= audio_norm)[0][0, :].unsqueeze(0)
        acc_audio_p2 = torchaudio.load(acc_p2, normalize= audio_norm)[0][0, :].unsqueeze(0)

        if fading:
            if i == 0:
                print("Fading Enabled")
            acc_audio_p1 = apply_fade(acc_audio_p1.squeeze(), sr = 16000)
            mic_audio_p1 = apply_fade(mic_audio_p1.squeeze(), sr = 16000)

            acc_audio_p2 = apply_fade(acc_audio_p2.squeeze(), sr = 16000)
            mic_audio_p2 = apply_fade(mic_audio_p2.squeeze(), sr = 16000)

            
        if preprocess:
            if i == 0:
                print("Preprocessing Enabled")
            #  Mic preprocess
            mic_audio_p1 = audio_preprocess(mic_audio_p1, accel_data= False)
            mic_audio_p2 = audio_preprocess(mic_audio_p2, accel_data= False)

            # Accel preprocess
            acc_audio_p1 = audio_preprocess(acc_audio_p1, accel_data= True)
            acc_audio_p2 = audio_preprocess(acc_audio_p2, accel_data= True)
        
        # Feature Extraction
        mic_p1_feats = extract_feature_Fbank(mic_audio_p1, n_MFCC = mic_MFCC).transpose(1, 2)
        accel_p1_feats = extract_feature_Fbank(acc_audio_p1, n_MFCC = acc_MFCC).transpose(1, 2)

        mic_p2_feats = extract_feature_Fbank(mic_audio_p2, n_MFCC = mic_MFCC).transpose(1, 2)
        accel_p2_feats = extract_feature_Fbank(acc_audio_p2, n_MFCC = acc_MFCC).transpose(1, 2)

        # print(mic_p1_feats.shape, accel_p1_feats.shape, mic_p2_feats.shape, accel_p2_feats.shape)
        if hstack:
            middle_padding = torch.zeros((mic_p1_feats.size(0), mic_p1_feats.size(1), hstack_frames_added), dtype = torch.float32)
            # print(mic_p1_feats.shape, middle_padding.shape, accel_p2_feats.shape)
            # fused_feats_p1 = torch.concat((mic_p1_feats, middle_padding, accel_p1_feats), dim = 2)
            fused_feats_p1 = torch.concat((mic_p1_feats[:, :, :301], middle_padding, accel_p1_feats[:, :, :301]), dim = 2)
            # print(fused_feats_p1.shape)
            # fused_feats_p2 = torch.concat((mic_p2_feats, middle_padding, accel_p2_feats), dim = 2)
            fused_feats_p2 = torch.concat((mic_p2_feats[:, :, :301], middle_padding, accel_p2_feats[:, :, :301]), dim = 2)
          
        else:
            # # Fusing
            # fused_feats_p1 = feats_fusion(mic_p1_feats, accel_p1_feats[:, :6, :], batch = False)
            # # print(fused_feats_p1.shape)
            # fused_feats_p2 = feats_fusion(mic_p2_feats, accel_p2_feats[:, :6, :], batch = False)
            if feature_mask:
                if i == 0:
                    print("Mic Masking Enabld")
                    # print("Random Gen enbaled")

                fused_feats_p1 = feats_fusion(torch.zeros_like(mic_p1_feats), accel_p1_feats, batch = False)
                fused_feats_p2 = feats_fusion(torch.zeros_like(mic_p2_feats), accel_p2_feats, batch = False)

                # fused_feats_p1 = feats_fusion(torch.randn_like(mic_p1_feats), accel_p1_feats, batch = False)
                # fused_feats_p2 = feats_fusion(torch.randn_like(mic_p2_feats), accel_p2_feats, batch = False)

            else:
            
                fused_feats_p1 = feats_fusion(mic_p1_feats, accel_p1_feats, batch = False)
                # print(fused_feats_p1.shape)
                fused_feats_p2 = feats_fusion(mic_p2_feats, accel_p2_feats, batch = False)

        # print(fused_feats_p1.shape, fused_feats_p2.shape)
        


        p1_emb = compute_embedding(mic_p1_feats, "embedding_model")
        p1_acc_emb = compute_embedding(accel_p1_feats, "acc_embedding_model")

        p2_emb = compute_embedding(mic_p2_feats, "embedding_model")
        p2_acc_emb = compute_embedding(accel_p2_feats, "acc_embedding_model")

        

        p1_full_emb = torch.concat((p1_emb, p1_acc_emb), dim = 2)
        p2_full_emb = torch.concat((p2_emb, p2_acc_emb), dim = 2)

        # w/ a fuser
        p1_full_emb = compute_embedding(p1_full_emb.transpose(1, 2), "embedding_fuser")
        p2_full_emb = compute_embedding(p2_full_emb.transpose(1, 2), "embedding_fuser")

        
        


        score = compute_cosine_similarity(p1_full_emb, p2_full_emb)
        all_scores.append(score)
        file.write(f"{mic_labels[i]} {p1} {p2} {score}\n")

        if i%1000==0:
            print(f"{i}/{pairs_len} pairs scored")

print("All pairs scored. Moving to EER calculation....=")


# Calculate EER 
EER, thresh = calculate_eer_and_curves(mic_labels, all_scores)

with open('missclassified_pairs.txt', 'w') as file:
    for j, (lab, score) in enumerate(zip(mic_labels, all_scores)):
        if lab == 1 and score < thresh:
            file.write(f"{lab} {score}, {mic_pths[j]}\n")
        elif lab == 0 and score > thresh:
            file.write(f"{lab} {score}, {mic_pths[j]}\n")



print(len(all_scores), EER, thresh)
# Best Epoch: 0.1368421052631579 0.7759395837783813 
# Last epoch :0.12973684210526315 0.7814710140228271


# 100 epochs
# 8600 0.12144736842105264 0.7338170409202576
# 8600 0.11105263157894738 0.7318519949913025



# New tests
#15170 0.175 0.7756084203720093
# ON ABCS VERIF PAIRS: 18530 0.144187675070028 0.8218458890914917