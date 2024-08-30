#
# Copyright (c) [2024] TDK U.S.A. Corporation
#

import os
import sys
import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from utils import *   






# Intitializing 
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

# Load hyperparameters file with command-line overrides
params_file, run_opts, overrides = sb.core.parse_arguments(sys.argv[1:])
with open(params_file) as fin:
    params = load_hyperpyyaml(fin, overrides)



# Create experiment directory
sb.core.create_experiment_directory(
    experiment_directory=params["output_folder"],
    hyperparams_to_save=params_file,
    overrides=overrides,
)

# run_on_main(params["pretrainer"].collect_files)


# Root for saving results
save_root = params['results_dir']
if not os.path.exists(save_root):
    os.makedirs(save_root)

# Loading the model
params["pretrainer"].load_collected()
params["embedding_model"].eval()
params["embedding_model"].to(run_opts["device"])



# Setting path variables
mic_pths, mic_labels = read_verif_file(params['data_path'], abs_path = params['is_path_abs'],
                                        base_path=params['base_verification_directory'])


# Pairs length and paths
pairs_len = len(mic_labels)
print(f"using the {params['data_path']} file with {pairs_len} pairs")


# Feature extraction config
verif_config = params['verif_config']
audio_norm = verif_config['audio_norm']
feat_mode = verif_config['feat_mode']
mic_MFCC = params['n_MFCC']

print('All Configs: ', verif_config)

# Starting the scoring process
all_scores = []

for i, (p1, p2) in enumerate(mic_pths):
    # First file
    mic_audio_p1 = read_audio(p1, normalize = audio_norm)
    
    # Second file
    mic_audio_p2 =  read_audio(p2, normalize = audio_norm)
    
    # Applying white gaussian noise with a specific SNR value
    if verif_config['WGN']:
        mic_audio_p2 = add_white_gaussian_noise(mic_audio_p2, verif_config['WGN_snr'])
        mic_audio_p1 = add_white_gaussian_noise(mic_audio_p1, verif_config['WGN_snr'])



    # Pre processing (gain + HP filter 35 Hz)  
    if verif_config['preprocess']:
        #  Mic preprocess
        mic_audio_p1 = audio_preprocess(mic_audio_p1, accel_data= False)
        mic_audio_p2 = audio_preprocess(mic_audio_p2, accel_data= False)


    # Feature Extraction
    mic_p1_feats = extract_feature_Fbank(mic_audio_p1, n_MFCC = mic_MFCC).transpose(1, 2)
    mic_p2_feats = extract_feature_Fbank(mic_audio_p2, n_MFCC = mic_MFCC).transpose(1, 2)





    # Just mic
    p1_emb = compute_embedding(mic_p1_feats, params["embedding_model"], device = run_opts["device"])
    p2_emb = compute_embedding(mic_p2_feats, params["embedding_model"], device = run_opts["device"])



    # Computing the cosine similarity score
    score = compute_cosine_similarity(p1_emb, p2_emb)
    all_scores.append(score)

    # Progress
    if i% params['print_freq'] ==0:
        print(f"{i}/{pairs_len} pairs scored")


print("All pairs scored. Moving to EER calculation....=")

# Calculate EER 
eer, eer_thresh = calculate_eer_and_curves(mic_labels, all_scores, save_root)

# Min-DCF Calcultions
min_dcf, dcf_thresh = calculate_minDCF(mic_labels, all_scores)


# Saving the pair scores and missclassified pairs 
missed_paths, missed_labels, missed_scores = extract_missclassified(all_scores, mic_labels, eer_thresh, mic_pths)
save_pairs_scores(missed_paths, missed_labels, missed_scores, save_root, missed = True)
save_pairs_scores(mic_pths, mic_labels, all_scores, save_root)




# Printing and saving the verification results
print(f'Number of pairs = {len(all_scores)}\n EER = {eer}, EER Threshold = {eer_thresh}\n min_DCF = {min_dcf}, min_DCF Threshold = {dcf_thresh}')
save_log(params['data_path'], params["output_folder"], eer, eer_thresh, min_dcf, dcf_thresh, save_root, len(all_scores))
