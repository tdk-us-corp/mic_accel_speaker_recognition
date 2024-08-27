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


# Root for saving results
save_root = params['results_dir']
if not os.path.exists(save_root):
    os.makedirs(save_root)

# Loading the model
params["pretrainer"].load_collected()
params["embedding_model"].eval()
params["embedding_model"].to(run_opts["device"])


# Setting path variables
mic_pths, mic_labels = read_verif_file(params['mic_path'], abs_path = params['is_path_abs'],
                                        base_path=params['base_verification_directory'])
acc_pths, acc_labels = read_verif_file(params['acc_path'], abs_path = params['is_path_abs'],
                                        base_path=params['base_verification_directory'])

# Pairs length and paths
pairs_len = len(mic_labels)
print(f"using the {params['mic_path']} and {params['acc_path']} files with {pairs_len} pairs")


# Feature extraction config
verif_config = params['verif_config']
audio_norm = verif_config['audio_norm']
input_feature_type = params['input_feature_type']
num_of_swapped_feats = verif_config['num_of_swapped_feats']
feat_mode = verif_config['feat_mode']
print(f"Modality: {input_feature_type}, Feature extraction mode: {feat_mode}")
mic_MFCC = params['mic_MFCC']
acc_MFCC = params['acc_MFCC']

print('All Configs: ', verif_config)


# Starting the scoring process
all_scores = []

for i, (p1, p2) in enumerate(mic_pths):

    # Sanity check the name of two audio files
    if verif_config['sanity_check']:
        if acc_pths[i][0].split("/")[-1][6:] != p1.split("/")[-1] or  acc_pths[+i][1].split("/")[-1][6:] != p2.split("/")[-1]:
            print("Warning! audio files might not be correctly matched")
    acc_p1 = acc_pths[i][0]
    acc_p2 = acc_pths[i][1]

    # reading the audio files (assuming the sr = 16KHz)
    # First file
    mic_audio_p1 = read_audio(p1, normalize = audio_norm)
    acc_audio_p1 = read_audio(acc_p1, normalize = audio_norm)
    
    # Second file
    mic_audio_p2 =  read_audio(p2, normalize = audio_norm)
    acc_audio_p2 = read_audio(acc_p2, normalize = audio_norm)
    
    # Applying white gaussian noise with a specific SNR value
    if verif_config['WGN']:
        mic_audio_p2 = add_white_gaussian_noise(mic_audio_p2, verif_config['WGN_snr'])
        mic_audio_p1 = add_white_gaussian_noise(mic_audio_p1, verif_config['WGN_snr'])

    # Fading in or Fading out effects
    if verif_config['fading']:
        acc_audio_p1 = apply_fade(acc_audio_p1.squeeze(), sr = 16000)
        mic_audio_p1 = apply_fade(mic_audio_p1.squeeze(), sr = 16000)

        acc_audio_p2 = apply_fade(acc_audio_p2.squeeze(), sr = 16000)
        mic_audio_p2 = apply_fade(mic_audio_p2.squeeze(), sr = 16000)

    # Pre processing (gain + HP filter 35 Hz)  
    if verif_config['preprocess']:
        # Swapping accel data with low pass of mic for spoofing attacks
        if verif_config['lowpass_attack']:
            acc_audio_p1 = mic_lowpass(mic_audio_p1)
            acc_audio_p2 = mic_lowpass(mic_audio_p2)         
                
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

    # Delta of first few coeffs fusion
    if verif_config['delta']:
        delta_coeffs = params['delta_coeffs']

        delta_feats_p1 =  mic_p1_feats[:, :delta_coeffs, :] - accel_p1_feats[:, :delta_coeffs, :]
        delta_feats_p2 = mic_p2_feats[:, :delta_coeffs, :] - accel_p2_feats[:, :delta_coeffs, :]

        mic_p1_feats = torch.concat((delta_feats_p1, mic_p1_feats[:, delta_coeffs:, :]), dim = 1)
        mic_p2_feats = torch.concat((delta_feats_p2, mic_p2_feats[:, delta_coeffs:, :]), dim = 1)

    # Swapping the first few coeffs fusion
    if verif_config['swap']:

        delta_feats_p1 =  accel_p1_feats[:, :num_of_swapped_feats, :]
        delta_feats_p2 = accel_p2_feats[:, :num_of_swapped_feats, :]

        mic_p1_feats = torch.concat((delta_feats_p1, mic_p1_feats[:, num_of_swapped_feats:, :]), dim = 1)
        mic_p2_feats = torch.concat((delta_feats_p2, mic_p2_feats[:, num_of_swapped_feats:, :]), dim = 1)



    # H-stacking the futures with zero padded columns in between
    if verif_config['h_stack']:
        middle_padding = torch.zeros((mic_p1_feats.size(0), mic_p1_feats.size(1), verif_config['hstack_frames_added']), dtype = torch.float32)
        fused_feats_p1 = torch.concat((mic_p1_feats, middle_padding, accel_p1_feats), dim = 2)
        # fused_feats_p1 = torch.concat((mic_p1_feats[:, :, :301], middle_padding, accel_p1_feats[:, :, :301]), dim = 2)
        fused_feats_p2 = torch.concat((mic_p2_feats, middle_padding, accel_p2_feats), dim = 2)
        # fused_feats_p2 = torch.concat((mic_p2_feats[:, :, :301], middle_padding, accel_p2_feats[:, :, :301]), dim = 2)
    
    # Masking microphone features with zeros or random values
    if verif_config['mic_feature_mask']:
        if verif_config['mask_random']:
            fused_feats_p1 = feats_fusion(torch.randn_like(mic_p1_feats), accel_p1_feats, batch = False)
            fused_feats_p2 = feats_fusion(torch.randn_like(mic_p2_feats), accel_p2_feats, batch = False)
        else:
            fused_feats_p1 = feats_fusion(torch.zeros_like(mic_p1_feats), accel_p1_feats, batch = False)
            fused_feats_p2 = feats_fusion(torch.zeros_like(mic_p2_feats), accel_p2_feats, batch = False)

        
    # Masking accel features with zeros or random values
    if verif_config['acc_feature_mask']:
        if verif_config['mask_random']:
            fused_feats_p1 = feats_fusion(mic_p1_feats, torch.zeros_like(accel_p1_feats), batch = False)
            fused_feats_p2 = feats_fusion(mic_p2_feats, torch.zeros_like(accel_p2_feats), batch = False)
        else:
            fused_feats_p1 = feats_fusion(mic_p1_feats, torch.randn_like(accel_p1_feats), batch = False)
            fused_feats_p2 = feats_fusion(mic_p2_feats, torch.randn_like(accel_p2_feats), batch = False)    
    


    if verif_config['v_stack']:

        # if the two audio files are not exactly aligned
        for_augmeneted_non_aligned = min(mic_p1_feats.size(-1), accel_p1_feats.size(-1))
        for_augmeneted_non_aligned_2 = min(mic_p2_feats.size(-1), accel_p2_feats.size(-1))

    
        try:
            fused_feats_p1 = feats_fusion(mic_p1_feats, accel_p1_feats, batch = False)
            fused_feats_p2 = feats_fusion(mic_p2_feats, accel_p2_feats, batch = False)
        except:
            print('non-aligned case detected')
            fused_feats_p1 = feats_fusion(mic_p1_feats[:, :, :for_augmeneted_non_aligned], accel_p1_feats[:, :, :for_augmeneted_non_aligned], batch = False)
            fused_feats_p2 = feats_fusion(mic_p2_feats[:, :, :for_augmeneted_non_aligned_2], accel_p2_feats[:, :, :for_augmeneted_non_aligned_2], batch = False)


    # Input feature type selection
    # Fused
    if input_feature_type == 'fused':
        p1_emb = compute_embedding(fused_feats_p1, params["embedding_model"], device = run_opts["device"])
        p2_emb = compute_embedding(fused_feats_p2, params["embedding_model"], device = run_opts["device"])

    # Just mic
    elif input_feature_type == 'mic':
        p1_emb = compute_embedding(mic_p1_feats, params["embedding_model"], device = run_opts["device"])
        p2_emb = compute_embedding(mic_p2_feats, params["embedding_model"], device = run_opts["device"])
        
    # Just acc
    elif input_feature_type == 'acc':
        p1_emb = compute_embedding(accel_p1_feats, params["embedding_model"], device = run_opts["device"])
        p2_emb = compute_embedding(accel_p2_feats, params["embedding_model"], device = run_opts["device"])
    


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
save_log(params['mic_path'], params['acc_path'], params["output_folder"], input_feature_type, eer, eer_thresh, min_dcf, dcf_thresh, save_root, len(all_scores))