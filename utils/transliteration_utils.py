import pandas as pd
import os

"""Loads the transliteration dataset
"""
def load_transliteration_dataset():

    dataset = pd.read_csv("data/external/transliteration/dataset.csv")

    known = dataset[dataset.from_source == True]
    dataset = dataset[["arabizi", "arabic", "from_source"]]

    dataset.columns = ["Arabize", "Arabic", "from_source"]

    #Store known words in a dict so we replace them later on instead of computing them with model again
    #This saves up computation time, and improves transliteration accuracy
    known = known[["arabizi", "arabic"]].set_index("arabizi", drop=True).arabic.to_dict()
    known_idx = list(known.keys())

    #Preprocess the text in the dataset
    dataset[["Arabize","Arabic"]] = dataset[["Arabize","Arabic"]].apply(preprocess_transliteration_text, axis=1)

    return dataset, (known, known_idx)

    
"""Preprocess Arabizi text before transliterating it into Arabic characters
"""
def preprocess_transliteration_text(text):
        
    x = text.copy()  
    
    def filter_letters_arabizi(word):
        
        word = word.replace("$", "s")
        word = word.replace("å", "a")
        word = word.replace("é", "e")
        word = word.replace("ê", "e")
        word = word.replace("ÿ", "y")
        word = word.replace("ą", "a")
        word = word.replace("ī", "i")
        word = word.replace("\n", "")
        word = word.replace("′", "'")
        
        return word
    
    x.Arabize = filter_letters_arabizi(str(x.Arabize))
    x.Arabic = x.Arabic
    
    return x