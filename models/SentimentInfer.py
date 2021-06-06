"""SentimentTrainer module

Loads pretrained models by the SentimentTrainer class for inference.

Example:

    from models import SentimentInfer
    from utils import load_sentiment_dataset

    train, test = load_sentiment_dataset()

    infer = SentimentInfer(folds=3, preprocess_function=lambda x:x, lang_prefix="en")
    probs = infer.infer(test["text"], return_probs=True)
    
    test["labels"] = SentimentInfer.Probs2Values(probs)

"""

import os
import yaml
import torch
import logging
import time

from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset, Sampler, SequentialSampler
from transformers import AutoTokenizer, AutoModel

from utils import preprocessing_for_bert, get_indices
from .bert_classifier import BertClassifier
from dataloader import BertDataset, data_collator, KSampler
from set_seed import set_seed


set_seed()
ALLOWED_LANGS = yaml.load(open("model_names.yaml"), Loader=yaml.FullLoader)["ALLOWED_LANGS"]

class SentimentInfer:

    def __init__(self, folds, preprocess_function, lang_prefix, checkpoint_folder="checkpoint", cpu=False
                    , maxlen=256, progress_bar=True):

        """Initialize instance of BertTrainer

        BertTrainer finetunes huggingface models for n number of folds on
        the sentiment analysis task. The models are later saved in `checkpoint_folder`
        directory.

        Args:
            folds (int): Number of folds used for traing
            preprocess_function (fn): Preprocessing function used during training for language
            lang_prefix (str): Language prefix, must exist in model_names.yaml
            checkpoint_folder (obj:pd.Series of int): Labels of the sentiment analysis dataset
            cpu (bool): Use CPU for inference
            maxlen (int): Maximum number of tokens in a single sentence
            progress_bar (bool): Show progress bar or not


        """
        
        self.checkpoint_folder = checkpoint_folder
        self.folds = folds
        self.lang_prefix = lang_prefix
        self.preprocess_function = preprocess_function
        self.maxlen = maxlen
        self.progress_bar = progress_bar

        #Verify that languages dict is valid
        assert lang_prefix in ALLOWED_LANGS, "Language prefix is invalid"

        if not cpu:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = "cpu"
        
        self.device = "cuda"
        self.models = [] 

        logging.info("Loading models ...")
        for i in range(folds):
            path = os.path.join(self.checkpoint_folder, lang_prefix + "_best_"+str(i))
            
            self.models.append(torch.load(path, map_location=self.device))
            logging.info(f"Model {lang_prefix} fold {i} loaded successfully")

        logging.info("All models are loaded")


        model_names = yaml.load(open("model_names.yaml"), Loader=yaml.FullLoader)
        self.model_name = model_names["model_name_" + lang_prefix]

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, do_lower_case=True)


    def bert_single_predict(self, model, test_dataloader):
        """Compute prediction for a single bert model
        """
        
        model.eval()
        all_logits = []

        if self.progress_bar:
            iterable = tqdm(test_dataloader)
        else: 
            iterable = test_dataloader

        for batch in iterable:

            b_input_ids, b_attn_mask = tuple(t.to(self.device) for t in batch)[:2]
            
            with torch.no_grad():
                logits = model(b_input_ids, b_attn_mask)
            all_logits.append(logits)
        
        all_logits = torch.cat(all_logits, dim=0)
        probs = F.softmax(all_logits, dim=1).cpu().numpy()

        return probs

    def bert_ensemble_predict(self, sentences, models, tokenizer, preprocess):
        """Preprocess sentence, and perform inference on n number of pretrained models
        """
        
        inputs, masks = preprocessing_for_bert(sentences, tokenizer, preprocess, max_len=self.maxlen)
        
        dataset = BertDataset(inputs, masks)
        sample = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sample, batch_size=128, collate_fn=data_collator)
        
        preds = []
        
        for model in models:
            preds.append(self.bert_single_predict(model, dataloader))
            
        return preds 


    def infer(self, text, return_probs=False):
        """Performs inference of a list of phrases
        """

        preds = self.bert_ensemble_predict(text, self.models, self.tokenizer, self.preprocess_function)
        out_sum = preds[0]
        for i in range(1,self.folds):
            out_sum = preds[i] + out_sum

        if return_probs:
            return out_sum/self.folds
        else:
            return out_sum.argmax()

    
    @staticmethod
    def Probs2Values(probs): 
        """Utility static function to convert probs back to their original form
        """

        assert probs.shape[1] == 3 , "Shape of probs array should be (*, 3)"
        preds = probs.argmax(1)

        preds = preds.where(preds == 2, 0)
        preds = preds.where(preds == 0, -1)
        
        return preds

        