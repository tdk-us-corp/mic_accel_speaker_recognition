# Config Instructions for the Training Phase

Some of the important variables to modify:

1. output_folder: Where the training log and environment variables will be stored
2. save_folder: Directory for saving the embedding model, classifier, optimizer, etc.
3. data_folder: The root directory of your database
4. train_annotation and valid_annotation: Path to the .csv files for training and validation
5. n_MFCC: The input feature dim of models(should be mic_MFCC + acc_MFCC for fusion models)
6. out_n_neurons: Number of speakers (classes) in your database for the classifier head
7. embedding_model: The config for the ECAPA TDNN backbone. For more info regarding the parameters: [ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification](https://arxiv.org/abs/2005.07143)




# Config Instructions for the Verification Phase

Some of the important variables to modify:
1. output_folder: path to your trained model. More specifically, you will need to use the directory where the embedding_modek.ckpt is stored
2. mic_path and accel_path: Path to the verification text files of each modality
3. results_dir: desired path for saving the output results which consist of 6 files:
    - missclassified_pairs.txt: Missclassified pair paths, labels and scores
    - all_pairs.txt: All of the pairs with their scores and labels
    - verification_log.txt: calculated metrics (EER, minDCF, etc.)
    - roc_curve.png, det_curve.png, frr_far_plot.png: Binary classification plots
4. n_MFCC: The input feature dim of models(should be mic_MFCC + acc_MFCC for fusion models)
5. embedding_model: **The config of the embedding model has to exactly match what you used during training**
6. input_feature_type: Selecting the verification mode (single-modality or fused)
