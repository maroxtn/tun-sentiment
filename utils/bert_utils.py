import re
import os

import pandas as pd

"""Get sentences ready for bert: Tokenize them, add special tokens
   Create mask and return in the form of a list

   Args:
    data (list): List of sentences to preprocess.
    tokenizer (object): The huggingface tokenizer.
    preprocess_text (fn): A custom function for preprocessing sentences, it takes only as string as input.
    max_len (int): After how many tokens to truncate sentence.

   Returns:
    input_ids (list): List of tokenized sentences.
    attention_masks (list): List of attention masks

   Note: Doesn't pad sentences because data collator will pad them later on
"""

def preprocessing_for_bert(data, tokenizer, preprocess_text, max_len=256):

    input_ids = []
    attention_masks = []
    tmp = tokenizer.encode("ab")[-1]

    for sentence in data:

        encoding = tokenizer.encode(preprocess_text(sentence))

        if len(encoding) > max_len:
            encoding = encoding[:max_len-1] + [tmp]

        in_ids = encoding
        att_mask = [1]*len(encoding)
        
        input_ids.append(in_ids)
        attention_masks.append(att_mask)

    return input_ids, attention_masks



"""Helper function to get multiple indexes from a list
"""
def get_indices(arr, idxs):  

    output = []
    for idx in idxs:
        output.append(arr[idx])
        
    return output



"""Preprocess function for English Bert model
"""
def en_bert_preprocess(text): 

    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    
    text = re.sub(r'([a-g-i-z][a-g-i-z])\1+', r'\1', text)
        
    return text



"""Load the sentiment analysis dataset in Pandas dataframe format
"""
def load_sentiment_dataset(directory="data/external/sentiment_analysis/", interim=False):

    if not interim:

        train = pd.read_csv(os.path.join(directory, "Train.csv"))
        test = pd.read_csv(os.path.join(directory, "Test.csv"))

        return train, test

    else:

        directory = directory.replace("external", "interim")

        train = pd.read_csv(os.path.join(directory, "Train.csv"))
        test = pd.read_csv(os.path.join(directory, "Test.csv"))

        return train, test  