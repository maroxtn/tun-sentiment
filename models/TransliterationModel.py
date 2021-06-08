"""TransliterationModel class

Trains / train and loads transliteration model, performs transliteration 
in batches. Most hyper-parameters are customizable. 

Load the dataset, train, and transliterate

Example:

    from models import TransliterationModel
    from utils import load_transliteration_dataset


    dataset, (known, known_idx) = load_transliteration_dataset()
    transliterate_model = TransliterationModel(dataset, load_weights=False, known=known, known_idx=known_idx)

    transliterate_model.train()
    transliterate_model.transliterate_phrase("Chbik ya m3alem")
    => شبيك يا معلم

Todo:
    * Include a notebook that runs everything (maybe on colab)
    * Make sure inference has the same piple (preprocess -> clean -> infer)
    
"""


import re
import logging
import os


import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm


from .optimizer import NoamOpt
from .transformer import TransformerModel, PositionalEncoding
from dataloader import Arab2ArabizDS, data_collator_Arab2Arabiz, seed_worker, KSampler
from set_seed import set_seed

set_seed()


"""Create the Transformer model that's used for transliteration
   Train it, and save it. Load it if needed.
"""

class TransliterationModel():

    def __init__(self, dataset, d_model=128, cpu=False, load_weights=False, checkpoint_folder="checkpoint", 
                    validation_frac=0.1, known=None, known_idx=None):

        """Init Transliteration class.

        Create a TransliterationModel instance, then either train your own model
        or load pretrained weights. transliterate_list, and transliterate_phrase
        both functions that use the model for transliteration
        
        Args:
            dataset (dataframe): The transliteration dataset as Pandas Dataframe.
            d_model (int): Dimension of the transformer model.
            cpu (bool): Use CPU or determine automatically.
            load_weights (bool): Wether to train the model or use a pre-trained one.
            checkpoint_folder (str): Directory of the model with "/".
            validation_frac (float): Fraction of the validation data. 
            known (dict): Dictionary of the known transliterated words. 
            known_idx (list): List of the index of the known transliterated words.
        """
        
        self.dataset = dataset
        self.validation_frac = validation_frac
        self.checkpoint_folder = checkpoint_folder
        self.d_model = d_model

        self.known = known
        self.known_idx = known_idx

        
        assert checkpoint_folder in os.listdir() + ["."] , f"Checkpoint folder :'{checkpoint_folder}' doesn't exist"

        #Compute max sequence length for input and output
        self.in_max = dataset.apply(lambda x: len(str(x.Arabize)), axis=1).max()
        self.out_max = dataset.apply(lambda x: len(x.Arabic), axis=1).max() + 2  #Take into account eos and sos

        pad_token = 0
        eos_token = 2
        sos_token = 1

        if not cpu:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = "cpu"

        logging.info(f"DEVICE: {self.device}")

        #Create the dictionary that maps each letter to it's corresponding embedding token
        #Input has one special token for padding
        in_tokens = set(" ".join(dataset.Arabize.values.tolist()).lower())
        self.in_token_to_int = {token: (i+1) for i,token in enumerate(sorted(in_tokens))}

        self.in_token_to_int[0] = "<pad>"

        #Out put has three special tokens, <eos> <sos> and <pad>
        out_tokens = set(" ".join(dataset.Arabic.values.tolist()))
        self.out_token_to_int = {token: (i+3) for i,token in enumerate(sorted(out_tokens))}

        self.out_token_to_int["<pad>"] = pad_token
        self.out_token_to_int["<sos>"] = sos_token
        self.out_token_to_int["<eos>"] = eos_token

        self.out_int_to_token = {self.out_token_to_int[t]:t for t in self.out_token_to_int}

        
        self.model = TransformerModel(len(self.in_token_to_int), len(self.out_token_to_int), d_model).to(self.device)
        logging.info(f"Model created! \n {self.model}")

        #Load model weights if pretrained
        if load_weights:
            self.model = torch.load(f"{checkpoint_folder}/transliterate", map_location=self.device)
            self.model = self.model.eval()
            logging.info(f"Model loaded from {checkpoint_folder}/transliterate")

    

    def tokenize_in(self, phrase, pad=True):
        """Tokenize Arabizi word and pad it to maxlen
        """

        tokenized = [self.in_token_to_int[i] for i in phrase.lower()]

        if pad:
            padded = tokenized + (self.in_max - len(tokenized)) * [self.out_token_to_int["<pad>"]] 
        else: padded = tokenized

        return padded


    def tokenize_out(self, phrase, pad=True):
        """Tokenize Arabic word and pad it to maxlen
        """

        tokenized = [self.out_token_to_int["<sos>"]] + [self.out_token_to_int[i] for i in phrase] + [self.out_token_to_int["<eos>"]]

        if pad:
            padded = tokenized + (self.out_max - len(tokenized)) * [self.out_token_to_int["<pad>"]]
        else: padded = tokenized

        return padded


    def tokenize_row(self, row):
        """Tokenize a row from the dataset (going to use it with apply method in dataframe)
        """

        x = row.copy()
        x.Arabize = self.tokenize_in(x.Arabize)
        x.Arabic = self.tokenize_out(x.Arabic)
        
        return x


    def tokenize_dataset(self):
        """Tokenize Arabizi and Arabic words in df
        """

        logging.info(f"Tokenizing {self.dataset.shape[0]} word pairs.")
        self.dataset = self.dataset.apply(lambda x: self.tokenize_row(x), axis=1)


    def train_validation_split(self):
        """Do Validation / Train split
        """

        validation = self.dataset.sample(frac=self.validation_frac)
        train = self.dataset.drop(validation.index)

        self.X_train = train.Arabize
        self.y_train = train.Arabic

        self.X_valid = validation.Arabize
        self.y_valid = validation.Arabic

        logging.info(f"Validation {len(self.X_valid)}\nTrain: {len(self.X_train)}")

    def getDataLoader(self, x, y, batch_size):
        """Get dataloader with its Sampler function, and data collator
        """

        data = Arab2ArabizDS(x, y)
        sampler = KSampler(data, batch_size)
        dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size, worker_init_fn=seed_worker, collate_fn=data_collator_Arab2Arabiz)

        return dataloader


    def run_epoch(self, iterator, optimizer, criterion):
        """Perform one training epoch
        """
        
        total_loss = 0

        for src, trg in iterator:

            src = src.T.to(self.device)
            trg = trg.T.to(self.device)

            output = self.model(src, trg[:-1, :])
            output = output.reshape(-1, output.shape[2])

            optimizer.optimizer.zero_grad()
            loss = criterion(output, trg[1:].reshape(-1))
            total_loss += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
            optimizer.step()


        return total_loss / len(iterator)

    def run_validation(self, iterator, criterion):
        """Get validation loss
        """

        total_loss = 0

        for src, trg in iterator:

            src = src.T.to(self.device)
            trg = trg.T.to(self.device)

            output = self.model(src, trg[:-1, :])
            output = output.reshape(-1, output.shape[2])


            loss = criterion(output, trg[1:].reshape(-1))
            total_loss += loss.item()

        return total_loss / len(iterator)


    def train_model(self, epochs=100, batch_size=32):
        """Train model for a specified number of epochs
        """

        logging.info("*"*30)
        logging.info("Starting the training of the transliteration model")
        logging.info(f"{epochs} epochs, {batch_size} batch_size")
        logging.info("*"*30)

        self.tokenize_dataset()
        self.train_validation_split()

        logging.info("Creating dataloader")
        train_dataloader = self.getDataLoader(self.X_train, self.y_train, batch_size)
        valid_dataloader = self.getDataLoader(self.X_valid, self.y_valid, batch_size)

        criterion = nn.CrossEntropyLoss(ignore_index=self.out_token_to_int["<pad>"])
        optimizer = NoamOpt(self.d_model, 1, 4000 ,optim.Adam(self.model.parameters(), lr=0))

        logging.info("Training started")
        min_loss = 99
        #Change model size 
        for i in range(epochs):
            
            loss = self.run_epoch(train_dataloader, optimizer, criterion)
            loss_val = self.run_validation(valid_dataloader, criterion)
            
            if loss_val < min_loss:
                min_loss = loss_val
                torch.save(self.model, self.checkpoint_folder + "/transliterate")

                logging.info("New best loss %f" % (min_loss))
            
            logging.info("EPOCH %d -- %f -- Val Loss: %f" % (i, loss, loss_val))

        logging.info("="*20)
        logging.info("Training done, best loss %f" % (min_loss))
        logging.info("="*20)

        self.model = torch.load(self.checkpoint_folder + "/transliterate").eval()


    def preprocess_text(self, text):
        """Preprocess incoming text for model
           Remove non allowed letters
        """

        text = text.replace('ß',"b")
        text = text.replace('à',"a")
        text = text.replace('á',"a")
        text = text.replace('ç',"c")
        text = text.replace('è',"e")
        text = text.replace('é',"e")
        text = text.replace('$',"s")
        text = text.replace("1","")
        
        
        text = text.lower()
        text = re.sub(r'[^A-Za-z0-9 ,!?.]', '', text)

        # Remove '@name'
        text = re.sub(r'(@.*?)[\s]', ' ', text)

        # Replace '&amp;' with '&'
        text = re.sub(r'&amp;', '&', text)

        # Remove trailing whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        text = re.sub(r'([h][h][h][h])\1+', r'\1', text)
        text = re.sub(r'([a-g-i-z])\1+', r'\1', text)  #Remove repeating characters
        text = re.sub(r' [0-9]+ ', " ", text)
        text = re.sub(r'^[0-9]+ ', "", text)

        return text

    
    #Keep numbers block
    def split(self, text):
        """Split sentences based on punctuation and spaces
           Store punctuation and known words (we don't need to predict words that exist in the dataset)

           Returns:
            Tuple: Splits of words to be passed through the model, and the removed words and their indexes
        """
        
        splits = re.findall(r"[\w']+|[?!.,]", text)

        to_be_added = []
        idx_to_be_added = []
        
        forbidden = ["?", "!", ".", ","] + self.known_idx

        for i, split in enumerate(splits):

            if split in forbidden:
                if split in self.known_idx:
                    to_be_added.append(self.known[split])
                else:
                    to_be_added.append(split)
                idx_to_be_added.append(i)

        splits = [i for i in splits if not i in forbidden]
        return splits, to_be_added, idx_to_be_added


    def transliterate_phrase(self, text):
        """Transliterate phrase into batches of word using greedy search

           Args:
            text (str): Sentence, or a group of sentences separated by a period.

           Returns:
            str: Splits of words to be passed through the model, and the removed words and their indexes
        """

        text = text.replace("0","")
        text = text.replace("6","")

        #Get splits
        text = self.preprocess_text(text)
        phrase, to_be_added, idx_to_be_added = self.split(text.lower())

        result = []

        #Sometimes all the words in a sentence exist in the known dict
        #So the returned phrase is empty, we check for that
        if len(phrase) > 0: 

            max_len_phrase = max([len(i) for i in phrase])

            #Pad and tokenize sentences
            input_sentence = []
            for word in phrase:
                input_sentence.append([self.in_token_to_int[i] for i in word] + [self.out_token_to_int["<pad>"]]*(max_len_phrase-len(word)))

            #Convert to Tensors
            input_sentence = torch.Tensor(input_sentence).long().T.to(self.device)
            preds = [[self.out_token_to_int["<sos>"]] * len(phrase)] 

            #A list of booleans to keep track of which sentences ended, and which sentences did not
            end_word = len(phrase) * [False]
            src_pad_mask = self.model.make_len_mask_enc(input_sentence)

            with torch.no_grad():

                src = self.model.pos_encoder(self.model.encoder(input_sentence))
                memory = self.model.transformer_encoder(src, None ,src_pad_mask)

                while not all(end_word): #Keep looping till all sentences hit <eos>
                    output_sentence = torch.Tensor(preds).long().to(self.device)

                    trg = self.model.pos_encoder(self.model.decoder(output_sentence))
                    output = self.model.transformer_decoder(tgt = trg, memory = memory, memory_key_padding_mask = src_pad_mask)
                    
                    output = self.model.fc_out(output)

                    output = output.argmax(-1)[-1].cpu().detach().numpy()
                    preds.append(output.tolist())

                    end_word = (output == self.out_token_to_int["<sos>"]) | end_word  #Update end word states
                    
                    if len(preds) > 50: #If word surpasses 50 characters, break out
                        break
                    
            preds = np.array(preds).T  #(words, words_len)

            for word in preds:  #De-tokenize predicted words
                tmp = []
                for i in word[1:]:   
                    if self.out_int_to_token[i] == "<eos>":
                        break
                    tmp.append(self.out_int_to_token[i])

                result.append("".join(tmp))
                
        #Re-add removed punctuation and words
        for item, idx in zip(to_be_added, idx_to_be_added):
            if item == "?":
                item = "؟"
            elif item == ",":
                item = "،"
            result.insert(idx, item)
                
        result = " ".join(result)
        return result


    def transliterate_list(self, texts, step_size=200, progress_bar=True):
        """Transliterate a list of phrases into batches of word using greedy search, then join them together

           Args:
            list: List of phrases in Arabizi.

           Returns:
            list: List of phrases converted into Arabic script
        """
        results = []
        if len(texts) < step_size:
            step_size = len(texts)

        if progress_bar:
            iterator = tqdm(range(0, len(texts), step_size))
        else:
            iterator = range(0, len(texts), step_size)

        for i in iterator: 
            
            out = self.transliterate_phrase(" lkrb3 ".join(texts[i:i+step_size]))
            splitted_sentences = [ex.strip() for ex in out.split(" " + self.transliterate_phrase("lkrb3") + " ")]

            if len(splitted_sentences) != len(texts[i:i+step_size]):
                logging.error("DANGER, a problem happened in de-tokenization, change the splitting word")
                break
            
            results.extend(splitted_sentences)

        return results
