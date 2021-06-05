import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset, Sampler
import numpy
import random

from set_seed import set_seed

set_seed()

"""Custom pytorch dataset for the sentiment analysis dataset
"""
class BertDataset(Dataset):

    def __init__(self, data, masks, label=None):
        
        self.data = data
        self.masks = masks
        
        if label != None:
            self.labels = label
        else:
            self.labels = None
        
        self.lengths = [len(i) for i in data]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.labels !=  None:
            return (self.data[idx], self.masks[idx], self.labels[idx], self.lengths[idx])
        else:  #For validation
            return (self.data[idx], self.masks[idx], None, self.lengths[idx])


"""Data collator for the bert model 
   Automatically pad batches to maxlen
   instead of padding to the maxlen of the
   datasete, this saves up significant computation
   and time.
"""
def data_collator(data):
    
    sentence, mask, label, length = zip(*data)
    
    tensor_dim = max(length)
    
    out_sentence = torch.full((len(sentence), tensor_dim), dtype=torch.long, fill_value=0)
    out_mask = torch.zeros(len(sentence), tensor_dim, dtype=torch.long)

    for i in range(len(sentence)):
        
        out_sentence[i][:len(sentence[i])] = torch.Tensor(sentence[i])
        out_mask[i][:len(mask[i])] = torch.Tensor(mask[i])
    
    if label[0] != None:
        return (out_sentence, out_mask, torch.Tensor(label).long())
    else:
        return (out_sentence, out_mask)