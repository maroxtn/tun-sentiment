import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset, Sampler
import numpy
import random

from set_seed import set_seed

set_seed()


"""Custom pytorch dataset for Arabizi to Arabic dataset
"""
class Arab2ArabizDS(Dataset):

    def __init__(self, data, label):
        
        self.data = data.values.tolist()
        self.labels = label.values.tolist()
        
        self.lengths_source = [len(i) for i in data]
        self.lengths_label = [len(i) for i in label]
        
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return (self.data[idx], self.labels[idx], self.lengths_source[idx], self.lengths_label[idx])


def data_collator_Arab2Arabiz(data):
    
    word, label, length_source, length_label = zip(*data)
    
    tensor_dim_1 = max(length_source)
    tensor_dim_2 = max(length_label)
    
    out_word = torch.full((len(word), tensor_dim_1), dtype=torch.long, fill_value=0)
    label_word = torch.full((len(word), tensor_dim_2), dtype=torch.long, fill_value=0)

    for i in range(len(word)):
        
        out_word[i][:len(word[i])] = torch.Tensor(word[i])
        label_word[i][:len(label[i])] = torch.Tensor(label[i])
    
    return (out_word, label_word)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


