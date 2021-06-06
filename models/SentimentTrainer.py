"""SentimentTrainer module

Trains a Sentiment Analysis n folds model.

Example:

    from model import SentimentTrainer
    from utils import load_sentiment_dataset

    #Very basic example , read the notes below on adding new language pretrained models
    train, test = load_sentiment_dataset()
    trainer = SentimentTrainer(train.text, train.labels, lambda x: x, "en", folds = 3)

    losses = trainer.train_all()
    print("\n".join([f"Fold {i} , Loss {val}" for i, val in enumerate(losses)]))

"""

import os
import yaml
import torch
import logging
import time


import numpy as np
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset, Sampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import KFold

from utils import preprocessing_for_bert, get_indices
from dataloader import BertDataset, data_collator, KSampler
from .bert_classifier import BertClassifier
from set_seed import set_seed, get_seed




set_seed()
ALLOWED_LANGS = yaml.load(open("model_names.yaml"), Loader=yaml.FullLoader)["ALLOWED_LANGS"]

class SentimentTrainer():

    def __init__(self, text, labels, preprocess_function, lang, folds=5, cpu=False, batch_size=32, 
                    dropout=0.05, n_epochs=4, lr= 5e-5, evaluate_step=200, checkpoint_folder="checkpoint", log_every=20):

        """Initialize instance of BertTrainer

        BertTrainer finetunes huggingface models for n number of folds on
        the sentiment analysis task. The models are later saved in `checkpoint_folder`
        directory.

        Notes:
            To add a new model language to be finetuned, for example Camembert
            you should add the prefix "fr" to ALLOWED_LANGS, then add huggingface
            camembert name (camembert-base) to model_names.yaml
            Naming convention in model_names.yaml: model_name_+language_abbr

        Args:
            text (obj:pd.Series of str): Pandas Series containing the sentences
            labels (obj:pd.Series of int): Labels of the sentiment analysis dataset
            preprocess_function (fn): Preprocessing function to be mapped on every sentence.
            lang (str`): Language prefix, must be one of the prefixes in ALLOWED_LANGS.
            cpu (bool): Use CPU or not.
            batch_size (int): Batch size used for training.
            dropout (float): Dropout for the clasifier
            n_epochs (int): Number of epochs.
            lr (float): Learning rate.
            evaluate_step (int): After how many training steps to evaluate.
                        note that evaluation only starts after the second epoch
                        evaluation during the first epochs would be pointless and
                        a waste of resources since the model didn't learn much.
            checkpoint_folder (str): Directory in which to save the models.
            log_every (int): Log training progress every `log_every` step.

        """
        
        #Neutral: 2, Negative: 0, Positive: 1. Will convert them back to original after inference
        self.labels = labels.replace(0,2).replace(-1,0).values.tolist()
        self.folds = folds
        self.batch_size = batch_size
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.lang = lang
        self.evaluate_step = evaluate_step
        self.checkpoint_folder = checkpoint_folder
        self.log_every = log_every

        assert checkpoint_folder in os.listdir() , f"Checkpoint folder :{checkpoint_folder}/ doesn't exist"
        assert lang in ALLOWED_LANGS, "Language should exist in config.yaml"
        
        if not cpu:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = "cpu"

        #TODO: Debug ^
        self.device = "cuda"
        #Load model name for the specified language from config yaml file
        model_names = yaml.load(open("model_names.yaml"), Loader=yaml.FullLoader)
        self.model_name = model_names["model_name_" + lang]

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, do_lower_case=True)
        self.preprocessed_data, self.masks = preprocessing_for_bert(text, self.tokenizer, preprocess_function, max_len=256)

    
    def instantiate_model(self, model_name, dataloader):
        """Instantiate a model and its corresponding sceduler and optimizer
        """

        bert_classifier = BertClassifier(model_name, dropout=self.dropout, freeze_bert=False)
        bert_classifier.to(self.device)

        optimizer = AdamW(bert_classifier.parameters(),
                        lr=self.lr,   
                        eps=1e-8 
                        )

        total_steps = len(dataloader) * self.n_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0, # Default value
                                                    num_training_steps=total_steps)

        return bert_classifier, optimizer, scheduler


    def train_fold(self, train_ids, val_ids, fold_id):
        """Preparing dataloader and dataset for a fold then train it
           Should not be used outside of class

           Args:
             train_ids (list): indexes of the training data
             val_ids (list): indexes of the validation data
             fold_id (int): fold number

           Returns:
             list: best loss value
        """

        logging.info(f"\n\n PREPARING FOLD NUMBER {fold_id}\n\n")

        X_train = get_indices(self.preprocessed_data, train_ids)
        y_train = get_indices(self.labels, train_ids)
        train_masks = get_indices(self.masks, train_ids)

        X_val = get_indices(self.preprocessed_data, val_ids)
        y_val = get_indices(self.labels, val_ids)
        val_masks = get_indices(self.masks, val_ids)

        X_val, y_val, val_masks = list(zip(*sorted(zip(X_val, y_val, val_masks), key=lambda x: len(x[0]))))  #Order the validation data for faster validation
        X_val, y_val, val_masks = list(X_val), list(y_val), list(val_masks)

        # Convert other data types to torch.Tensor
        y_train = torch.tensor(y_train)
        y_val = torch.tensor(y_val)

        # Create the DataLoader for our training set
        train_data = BertDataset(X_train, train_masks, y_train)
        train_sampler = KSampler(train_data, self.batch_size)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.batch_size, collate_fn=data_collator)

        # Create the DataLoader for our validation set
        val_data = BertDataset(X_val, val_masks, y_val)
        val_sampler = SequentialSampler(val_data)
        val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=self.batch_size, collate_fn=data_collator)
        
        logging.info(f"\n\nTRAININF FOLD NUMBER {fold_id}\n\n")
        bert_classifier, optimizer, scheduler = self.instantiate_model(self.model_name, train_dataloader)

        return self.train(bert_classifier, train_dataloader, optimizer, scheduler,
                     val_dataloader, epochs=self.n_epochs, evaluation=True, fold=fold_id, prefix=self.lang)


    def train(self, model, train_dataloader, optimizer, scheduler,
                     val_dataloader=None, epochs=4, evaluation=False, fold=0, prefix=""):

        """Trains a model, and logs the training data
        """
        
        max_acc = -99
        loss_fn = nn.CrossEntropyLoss()

        logging.info("Start training...\n")
        for epoch_i in range(epochs):

            logging.info(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
            logging.info("-"*70)

            t0_epoch, t0_batch = time.time(), time.time()

            total_loss, batch_loss, batch_counts = 0, 0, 0
            model.train()

            for step, batch in enumerate(train_dataloader):
                batch_counts +=1

                b_input_ids, b_attn_mask, b_labels = tuple(t.to(self.device) for t in batch)

                model.zero_grad()

                logits = model(b_input_ids, b_attn_mask)

                loss = loss_fn(logits, b_labels)
                batch_loss += loss.item()
                total_loss += loss.item()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()

                if (step % self.log_every == 0 and step != 0) or (step == len(train_dataloader) - 1):

                    time_elapsed = time.time() - t0_batch

                    logging.info(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                    batch_loss, batch_counts = 0, 0
                    t0_batch = time.time()
                    
                if step%self.evaluate_step == 0 and step != 0 and epoch_i != 0 and epoch_i != 1: #Only start evaluating every 200 steps after the second epoch
                    
                    logging.info("-"*70)

                    if evaluation == True:

                        val_loss, val_accuracy = self.evaluate(model, val_dataloader, loss_fn)
                        
                        if val_accuracy > max_acc:
                            max_acc = val_accuracy
                            torch.save(model, os.path.join(self.checkpoint_folder, prefix+"_best_"+str(fold)))
                            logging.info("new max")
                            

                        logging.info(val_accuracy)
                        
                        logging.info("-"*70)
                    logging.info("\n")
                    
                    model.train()

            avg_train_loss = total_loss / len(train_dataloader)

            logging.info("-"*70)

            if evaluation == True:
                
                val_loss, val_accuracy = self.evaluate(model, val_dataloader, loss_fn)
                
                if val_accuracy > max_acc:
                    max_acc = val_accuracy
                    torch.save(model, os.path.join(self.checkpoint_folder, prefix+"_best_"+str(fold)))
                    logging.info("new max")

                time_elapsed = time.time() - t0_epoch
                
                logging.info(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
                logging.info("-"*70)
            logging.info("\n")
        
        logging.info("Fold Training complete!")
        return max_acc


    def evaluate(self, model, val_dataloader, loss_fn):
        """Evaluate the trained model
        """

        model.eval()

        val_accuracy = []
        val_loss = []

        for batch in val_dataloader:

            b_input_ids, b_attn_mask, b_labels = tuple(t.to(self.device) for t in batch)

            with torch.no_grad():
                logits = model(b_input_ids, b_attn_mask)

            loss = loss_fn(logits, b_labels)
            val_loss.append(loss.item())

            preds = torch.argmax(logits, dim=1).flatten()

            accuracy = (preds == b_labels).cpu().numpy().mean() * 100
            val_accuracy.append(accuracy)

        val_loss = np.mean(val_loss)
        val_accuracy = np.mean(val_accuracy)

        return val_loss, val_accuracy



    def train_all(self):
        """Train all folds
        """

        kfold = KFold(self.folds, True, get_seed())
        fold_id = 0

        bests = []

        for train_ids, val_ids in kfold.split(self.preprocessed_data):
            max_loss = self.train_fold(train_ids, val_ids, fold_id)
            bests.append(max_loss)

            fold_id += 1

        return bests