"""

Trains sentiment analysis models using command line arguments (CLI)

Example:

    python train_sentiment.py en --epochs 4 --batch_size 16 --save_folder .

"""

import torch

import random
import numpy as np
import pandas as pd
import yaml

import os
import logging
import argparse

from utils import load_sentiment_dataset, en_bert_preprocess
from models import SentimentTrainer
from set_seed import set_seed



logging.basicConfig(format='%(asctime)s %(message)s: ', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
ALLOWED_LANGS = yaml.load(open("model_names.yaml"), Loader=yaml.FullLoader)["ALLOWED_LANGS"]
set_seed()

def main():

    parser = argparse.ArgumentParser(description="Train the sentiment analysis models,\
                                         if there is any models already trained, this command would replace them.")


    parser.add_argument("lang_prefix", help="Language prefix of the huggingface pretrained model. Check model_names.yaml.",
                    choices=ALLOWED_LANGS)

    parser.add_argument("--epochs", help="Number for epochs. Default: 4", default=4, type=int)
    parser.add_argument("--batch_size", help="Batch size. Default: 32", default=32, type=int)
    parser.add_argument("--evaluate_every", help="Evaluation every n steps", default=200, type=int)
    parser.add_argument("--save_folder", help="Folder to save model in. `checkpoint` is default value", default="checkpoint")
    parser.add_argument("--folds", help="Number of Kfolds used in training.", default=5, type=int)
    parser.add_argument("--dropout", help="Dropout value.", default=0.05, type=float)
    parser.add_argument("--lr", help="Learning rate.", default=5e-5, type=float)
    parser.add_argument("--log_every", help="Log every n step.", default=20, type=int)
    parser.add_argument("--dataset_dir", help="Transliteration dataset directory. Default: `data/external/sentiment_analysis`"
            ,default="data/external/sentiment_analysis")


    args = parser.parse_args()
    

    train, test = load_sentiment_dataset(directory=args.dataset_dir, interim=True)

    #Load the appropriate sentences and preprocessing function depending on the language
    if args.lang_prefix == "en":
        fn = en_bert_preprocess
        text = train.text
    else:
        fn = lambda x: x
        text = train.text_arabic

    models = SentimentTrainer(text, train.label, fn, args.lang_prefix, args.folds, args.batch_size
                    ,dropout=args.dropout, n_epochs=args.epochs, lr=args.lr, evaluate_step=args.evaluate_every
                    ,checkpoint_folder=args.save_folder, log_every=args.log_every)

    losses = models.train_all()
    logging.info("\n".join([f"Fold {i} , Loss {val}" for i, val in enumerate(losses)]))


if __name__ == "__main__":
    main()
