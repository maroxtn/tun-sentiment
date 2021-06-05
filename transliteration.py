"""Transliteration module

Trains / train and loads transliteration model, performs transliteration 
in batches. Most hyper-parameters are customizable. Running this module
in main would transliterate the sentiment analysis dataset and store it
in data/iterim/Train.csv and data/interim/Test.csv

Example:

    Load the dataset, train, and transliterate

    import transliteration

    dataset, (known, known_idx) = transliteration.load_transliteration_dataset()
    transliterate_model = transliteration.TransliterationModel(dataset, load_weights=False, known=known, known_idx=known_idx)

    transliterate_model.train()
    transliterate_model.transliterate_phrase("Chbik ya m3alem")
    => شبيك يا معلم

    Todo:
        * Impelement Sentiment Analysis module
        * Include a notebook that runs everything (maybe on colab)
        * Make sure inference has the same piple (preprocess -> clean -> infer)
        * Maybe abstract it away to contain any possible model language ? 
        * Add a controller to load the models and perform inference on them
        * Add a main file to use the class
        * Verify if Documentation is written properly 

Ideas of project:

	- Fairseq m2m for translation fine tuning
	- 

-------------
See if i can contribute in fairseq
Check projects for contribution
Read pep
Create repo for sentiment analysis project and put it on github
Send mail to icompaass and vermeg
Also see if there is a paper you can implement

"""

import torch

import random
import numpy as np
import pandas as pd

import os
import logging

from utils import load_transliteration_dataset
from models import TransliterationModel
from set_seed import set_seed

logging.basicConfig(level=logging.INFO)


def main():

    set_seed()

    dataset, (known, known_idx) = load_transliteration_dataset()

    transliterate_model = TransliterationModel(dataset, load_weights=True, known=known, known_idx=known_idx)
    
    sentiment_train = pd.read_csv("data/external/sentiment_analysis/Train.csv")
    sentiment_test = pd.read_csv("data/external/sentiment_analysis/Test.csv")

    sentiment_train["text_arabic"] = transliterate_model.transliterate_list(sentiment_train.text)
    sentiment_test["text_arabic"] = transliterate_model.transliterate_list(sentiment_test.text)

    sentiment_train.to_csv("data/interim/train.csv")
    sentiment_test.to_csv("data/interim/test.csv")


if __name__ == "__main__":
    main()
