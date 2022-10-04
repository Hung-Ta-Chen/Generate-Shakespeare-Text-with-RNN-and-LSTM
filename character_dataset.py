# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 00:51:26 2021

@author: narut
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset

class CharacterDataset(Dataset):
  def __init__(self, text, seq_len = 100, char2idx = None, idx2char = None, char_set = None):
    self.text = text
    self.seq_len = seq_len
    
    # Character's collection
    if char_set == None:
      chars = set(text)
    else:
      chars = char_set
    
    self.char_set = list(chars)
    
    # Construct character's dictionary
    if char2idx == None:
      self.char2idx = {c:i for i ,c in enumerate(chars)}
    else:
      self.char2idx = char2idx
    
    if idx2char == None:
      self.idx2char = dict(enumerate(chars))
    else:
      self.idx2char = idx2char
    
    
  def __len__(self):
    return len(self.text) - self.seq_len
  
  def __getitem__(self, idx):
    X = torch.LongTensor(
            [self.char2idx[c] for c in self.text[idx : idx + self.seq_len]]
        )
    y = self.char2idx[self.text[idx + self.seq_len]]

    return X, y
