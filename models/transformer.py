import torch

from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer
import torch.nn.functional as F

import torch.nn as nn

import math
from set_seed import set_seed

set_seed()


"""Create positional embedding: https://nlp.seas.harvard.edu/2018/04/03/attention.html
"""
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=9000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = nn.Parameter(torch.ones(1))

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.scale * self.pe[:x.size(0), :]
        return self.dropout(x)



"""Create a standard transformer model

Args:
    intoken, outtoken (int): Number of tokens in both input and output text
    hidden (int): Dimension of the model (d_model)
    enc_layers, dec_layers (int): Encoder and decoder layer count
    dropout: dropout
    nheads: Number of attention heads
    pad_token: The padding token 
"""
class TransformerModel(nn.Module):
    
    def __init__(self, intoken, outtoken ,hidden, enc_layers=1, dec_layers=1, dropout=0.15, nheads=4, pad_token=0):
        super(TransformerModel, self).__init__()
        
        ff_model = hidden*4
        self.pad_token = pad_token
        
        self.encoder = nn.Embedding(intoken, hidden)
        self.pos_encoder = PositionalEncoding(hidden, dropout)

        self.decoder = nn.Embedding(outtoken, hidden) 
        self.pos_decoder = PositionalEncoding(hidden, dropout)
        
        
        encoder_layers = TransformerEncoderLayer(d_model=hidden, nhead = nheads, dim_feedforward = ff_model, dropout=dropout, activation='relu')
        self.transformer_encoder = TransformerEncoder(encoder_layers, enc_layers)

        encoder_layers = TransformerDecoderLayer(hidden, nheads, ff_model, dropout, activation='relu')
        self.transformer_decoder = TransformerDecoder(encoder_layers, dec_layers)        

        self.fc_out = nn.Linear(hidden, outtoken)

        self.src_mask = None
        self.trg_mask = None
        self.memory_mask = None

        
    """Triangular mask for attention: https://arxiv.org/pdf/1706.03762.pdf
    """
    def generate_square_subsequent_mask(self, sz, sz1=None):
        
        if sz1 == None:
            mask = torch.triu(torch.ones(sz, sz), 1)
        else:
            mask = torch.triu(torch.ones(sz, sz1), 1)
            
        return mask.masked_fill(mask==1, float('-inf'))

    """Create padding mask
    """
    def make_len_mask_enc(self, inp):
        return (inp == self.pad_token).transpose(0, 1)   #(batch_size, output_seq_len)
    
    def make_len_mask_dec(self, inp):
        return (inp == self.pad_token).transpose(0, 1) #(batch_size, input_seq_len)
    


    def forward(self, src, trg): #SRC: (seq_len, batch_size)

        #Subsequent mask
        if self.trg_mask is None or self.trg_mask.size(0) != len(trg):
            self.trg_mask = self.generate_square_subsequent_mask(len(trg)).to(trg.device)
            

        #Adding padding mask
        src_pad_mask = self.make_len_mask_enc(src)
        trg_pad_mask = self.make_len_mask_dec(trg)
             

        #Add embeddings Encoder
        src = self.encoder(src)  #Embedding, (seq_len, batch_size, d_model)
        src = self.pos_encoder(src)   #Pos embedding
        
        
        #Add embedding decoder
        trg = self.decoder(trg) #(seq_len, batch_size, d_model)
        trg = self.pos_decoder(trg)

        
        memory = self.transformer_encoder(src, None, src_pad_mask)
        output = self.transformer_decoder(tgt = trg, memory = memory, tgt_mask = self.trg_mask, memory_mask = None, 
                                          tgt_key_padding_mask = trg_pad_mask, memory_key_padding_mask = src_pad_mask)

        output = self.fc_out(output)

        return output