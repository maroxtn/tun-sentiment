"""  

CLI interface for sentences transliteration in data/external/['train', 'test'].csv to 
data/interim/['train', 'test'].csv. No arguments needed.

Example:

    python transliteration_inference.py

"""

import os
import logging
import argparse

import torch
import random
import numpy as np
import pandas as pd

from utils import load_sentiment_dataset, load_transliteration_dataset
from models import TransliterationModel
from set_seed import set_seed



logging.basicConfig(format='%(asctime)s %(message)s: ', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
set_seed()

def main():
    
    parser = argparse.ArgumentParser(description="Transliterate sentences in data/external/['train', 'test'].csv to \
                                data/interim/['train', 'test'].csv. No arguments needed.")
    args = parser.parse_args()


    #Prepare transliterated data
    dataset, (known, known_idx) = load_transliteration_dataset()
    transliterate_model = TransliterationModel(dataset, load_weights=True, known=known, known_idx=known_idx)



    train, test = load_sentiment_dataset()

    train["text_arabic"] = transliterate_model.transliterate_list(train.text)
    test["text_arabic"] = transliterate_model.transliterate_list(test.text)

    train.to_csv("data/interim/train.csv")
    test.to_csv("data/interim/test.csv")


if  __name__ == "__main__":
    main()

