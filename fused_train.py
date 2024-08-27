"""Inspired by the speechBrain's script for training speaker recognition models:
https://github.com/speechbrain/speechbrain/blob/develop/recipes/VoxCeleb/SpeakerRec/train_speaker_embeddings.py
"""

import os
import sys
import random
import torch
import torchaudio
import speechbrain as sb
from speechbrain.utils.data_utils import download_file
from hyperpyyaml import load_hyperpyyaml
from speechbrain.processing.features import DCT
from torch.distributions import Bernoulli






class SpeakerBrain(sb.core.Brain):
    def __init__(self, modules=None, opt_class=None, hparams=None, run_opts=None, checkpointer=None):
        super().__init__(modules, opt_class, hparams, run_opts, checkpointer)

        # MFCC config
        self.is_MFCC = True
        self.mic_MFCC = hparams['mic_MFCC']
        self.acc_MFCC = hparams['acc_MFCC']

    
    """Speaker embedding training"""
    def compute_forward(self, batch, stage):

        batch = batch.to(self.device)
        wavs, lens = batch.sig

        # Computing the features of each modality separately, wavs shape : (batch_size, batch_audio_sample_length, 2)
        mic_feats = self.modules.compute_features(wavs[:, :, 0])
        accel_feats = self.modules.compute_features(wavs[:, :, 1])

        
        # Applying DCT to melbanks to extract mfccs
        if self.is_MFCC:
            mic_feats = self.compute_DCT(mic_feats, num_MFCC = self.mic_MFCC)
            accel_feats = self.compute_DCT(accel_feats, num_MFCC = self.acc_MFCC)

        # Fusing the input features
        all_feats = torch.concat((mic_feats, accel_feats), dim = 2)


        # Mean & var normalization
        final_feats = self.modules.mean_var_norm(all_feats, lens)

        # Computing the embeddings
        embeddings = self.modules.embedding_model(final_feats)
        
        # Classification had
        outputs = self.modules.classifier(embeddings)

        return outputs, lens
    
    def compute_DCT(self, feats, num_MFCC):
        """DCT calculation for MFCC as the speechbrain function had some problems"""
        compute_dct = DCT(feats.size(-1), n_out = num_MFCC)
        dct_feats = compute_dct(feats)

        return dct_feats
    
    
    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss using speaker-id as label."""
        predictions, lens = predictions
        uttid = batch.id
        spkid, _ = batch.spk_id_encoded


        loss = self.hparams.compute_cost(predictions, spkid, lens)

        if stage == sb.Stage.TRAIN and hasattr(
            self.hparams.lr_annealing, "on_batch_end"
        ):
            self.hparams.lr_annealing.on_batch_end(self.optimizer)

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(uttid, predictions, spkid, lens)

        return loss



    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of an epoch."""
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["ErrorRate"] = self.error_metrics.summarize("average")

        # Perform end-of-iteration step
        if stage == sb.Stage.VALID:
            # cyclic LR 
            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"ErrorRate": stage_stats["ErrorRate"]},
                min_keys=["ErrorRate"],
            )


def dataio_prep(hparams):
    "Data processing pipelines"

    # Dataset root folder
    data_folder = hparams["data_folder"]


    # 1. Declarations:
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_annotation"],
        replacements={"data_root": data_folder},
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_annotation"],
        replacements={"data_root": data_folder},
    )
    data_root = hparams['data_folder']

    datasets = [train_data, valid_data]
    label_encoder = sb.dataio.encoder.CategoricalEncoder()


    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav", "start", "stop", "duration")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, start, stop, duration):
        
        # Separating modalities paths 
        mic_wav, accel_wav = wav.split(",")
        mic_wav = os.path.join(data_root, mic_wav)
        accel_wav = os.path.join(data_root, accel_wav[1:])

        # Extracting the segment
        start = int(start)
        stop = int(stop)
        num_frames = stop - start

        mic_sig, _ = torchaudio.load(
            mic_wav, num_frames=num_frames, frame_offset=start
        )
        accel_sig, _ = torchaudio.load(
            accel_wav, num_frames=num_frames, frame_offset=start
        )
        mic_sig = mic_sig.transpose(0, 1)
        accel_sig = accel_sig.transpose(0, 1)
        
        try:
            all_sigs = torch.cat((mic_sig, accel_sig), dim = 1)

        # Dealing with possible sliced accel audio
        except RuntimeError:
            min_len = max(mic_sig.shape[0], accel_sig.shape[0])
            all_sigs = torch.cat((mic_sig[:min_len,:], accel_sig[:min_len,:]), dim = 1)
            print(mic_wav, accel_wav)
            print(mic_sig.shape, accel_sig.shape)
            print(start, stop)

        # sanity check, should be True False True False
        # print(torch.equal(all_sigs[:, 0], mic_sig.squeeze(1)))
        # print(torch.equal(all_sigs[:, 0], accel_sig.squeeze(1)))
        # print(torch.equal(all_sigs[:, 1], accel_sig.squeeze(1)))
        # print(torch.equal(all_sigs[:, 1], mic_sig.squeeze(1)))
        return all_sigs

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("spk_id")
    @sb.utils.data_pipeline.provides("spk_id", "spk_id_encoded")
    def label_pipeline(spk_id):
        yield spk_id
        spk_id_encoded = label_encoder.encode_sequence_torch([spk_id])
        yield spk_id_encoded

    sb.dataio.dataset.add_dynamic_item(datasets, label_pipeline)

    # 3. Fit encoder:
    # Load or compute the label encoder (with multi-GPU DDP support)
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[train_data],
        output_key="spk_id",
    )

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "spk_id_encoded"])

    return train_data, valid_data, label_encoder


if __name__ == "__main__":

    # This flag enables the inbuilt cudnn auto-tuner
    torch.backends.cudnn.benchmark = True

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    
    train_data, valid_data, label_encoder = dataio_prep(hparams)

    # Create experiment directory
    sb.core.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Brain class initialization
    speaker_brain = SpeakerBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    if speaker_brain.is_MFCC:
        print("using MFCC instead of MelSpec")

    # Training
    speaker_brain.fit(
        speaker_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )