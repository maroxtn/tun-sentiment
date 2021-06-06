"""  

Perform inference on test dataframe and exports results to data/final/test.csv

Example:

    python sentiment_inference.py --ar_folds 5 --en_folds 5

"""

import os
import logging
import argparse

import torch
import random
import numpy as np
import pandas as pd

from utils import load_sentiment_dataset, load_transliteration_dataset
from models import SentimentInfer, SentimentTrainer
from utils import en_bert_preprocess
from models import TransliterationModel
from set_seed import set_seed



logging.basicConfig(format='%(asctime)s %(message)s: ', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
set_seed()

def main():

    parser = argparse.ArgumentParser(description="Perform inference on test dataframe and exports results to data/final/test.csv")

    parser.add_argument("--ar_folds", help="Number of folds used while training ar bert model.", default=5, type=int)
    parser.add_argument("--en_folds", help="Number of folds used while training bert-base.", default=5, type=int)

    args = parser.parse_args()



    train, test = load_sentiment_dataset(interim=True)


    enInfer = SentimentInfer(folds=args.ar_folds, preprocess_function=en_bert_preprocess, lang_prefix="en")
    arInfer = SentimentInfer(folds=args.en_folds, preprocess_function=lambda x:x, lang_prefix="ar")

    probs_en = enInfer.infer(test["text"], return_probs=True)
    probs_ar = arInfer.infer(test["converted"], return_probs=True)

    probs = probs_en+probs_ar*1.30
    predictedLabels = SentimentInfer.Probs2Values(probs)

    #Export predicted data
    test["labels"] = predictedLabels
    test[["ID", "labels"]].to_csv("data/final/test.csv")

    print("Produced file exported to data/final/Test.csv")


if  __name__ == "__main__":
    main()

