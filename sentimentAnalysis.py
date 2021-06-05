import torch

import random
import numpy as np
import pandas as pd

import os
import logging

import transliteration
from models import SentimentInfer, SentimentTrainer
from utils import en_bert_preprocess

logging.basicConfig(level=logging.INFO)


def main():
    

    #Prepare transliterated data
    dataset, (known, known_idx) = transliteration.load_transliteration_dataset()
    transliterate_model = transliteration.TransliterationModel(dataset, load_weights=False, known=known, known_idx=known_idx)

    transliterate_model.train()


    sentiment_train = pd.read_csv("data/external/sentiment_analysis/Train.csv")
    sentiment_test = pd.read_csv("data/external/sentiment_analysis/Test.csv")

    sentiment_train["text_arabic"] = transliterate_model.transliterate_list(sentiment_train.text)
    sentiment_test["text_arabic"] = transliterate_model.transliterate_list(sentiment_test.text)

    sentiment_train.to_csv("data/interim/train.csv")
    sentiment_test.to_csv("data/interim/test.csv")

    logging.info("Converted Arabizi dataset into Arabic script")


    #Fine tune models
    en_model = SentimentTrainer(sentiment_train.text, sentiment_train.labels, en_bert_preprocess, "en")
    en_losses = en_model.train_all()

    logging.info(f"\n\n\nBert-base training is done")
    logging.info(f"Losses: {str(en_losses)}\n\n\n")

    ar_model = SentimentTrainer(sentiment_train.text_arabic, sentiment_train.labels, lambda x:x, "ar")
    ar_model = ar_model.train_all()   


    #Perform inference
    enInfer = SentimentInfer(folds=5, preprocess_function=en_bert_preprocess, lang_prefix="en")
    arInfer = SentimentInfer(folds=5, preprocess_function=lambda x:x, lang_prefix="ar")

    probs_en = enInfer.infer(sentiment_test["text"], return_probs=True)
    probs_ar = arInfer.infer(sentiment_test["text_arabic"], return_probs=True)

    probs = probs_en+probs_ar*1.30
    predictedLabels = SentimentInfer.Probs2Values(probs)


    #Export predicted data
    sentiment_test["labels"] = predictedLabels
    sentiment_test[["ID", "labels"]].to_csv("data/final/test.csv")

if  __name__ == "__main__":
    main()

