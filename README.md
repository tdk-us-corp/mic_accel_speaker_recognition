# Requirements

```bash
pip install -r requirements.txt
```


# Training a model

To train a fusion model, you can follow the following steps:
1. Creating a new .csv training and validation file using the preprocessing/splitting_fused.py script (or using the already created ones in the datasets folder %TO DO ADD IT HERE)
2. Updating the config file accordingly at configs/train_ecapa_MFCC.yaml (configs/config_instructions.md for more info)
3. Running the following code:

```python
python3 fused_train.py  train_configs/train_ecapa_MFCC_fused.yaml
```

Note: You can also train a single-modality model by using the following command:
```python
python3 train.py  train_configs/train_ecapa_MFCC.yaml
```

# Speaker Verification Inference

To evaluate your trained speaker verification models, you can follow the following steps (if you want to just use the provided pairs, skip steps 1 and 2):
1. Creating your own speaker verification pairs by using the preprocessing/createVerificationPairsSeparateEars.py for microphone data
2. Using the verif_pairs/accelPairsFromMic.py to create the accel verification file from the microphone version
3. Updating the config file accordingly at configs/verification_ecapa_MFCC_fused.yaml (configs/config_instructions.md for more info)
4. Running the following code:

```python
python3 speaker_verification_cosine_fused.py configs/verification_ecapa_MFCC_fused.yaml
```
Note: You can also evaluate the single modality models with the following config by passing the correct input_feature_type argument in the .yaml file.




