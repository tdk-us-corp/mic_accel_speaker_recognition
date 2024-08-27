import os
import sys
import torch
import logging
import speechbrain as sb
from tqdm.contrib import tqdm
from hyperpyyaml import load_hyperpyyaml




if __name__ == "__main__":

    # Load hyperparameters file with command-line overrides
    params_file, run_opts, overrides = sb.core.parse_arguments(sys.argv[1:])
    with open(params_file) as fin:
        params = load_hyperpyyaml(fin, overrides)

    # Loading the embedding model
    params["pretrainer"].load_collected()
    params["embedding_model"].eval()
    params["embedding_model"].to(run_opts["device"])


    print("\nType of the loaded model: \n ", type(params["embedding_model"]))
    def count_parameters(model):
       return sum(p.numel() for p in model.parameters())

    total_params = count_parameters(params["embedding_model"])
    print(f'\nTotal number of parameters: {total_params}')

    # print(params["embedding_model"])


