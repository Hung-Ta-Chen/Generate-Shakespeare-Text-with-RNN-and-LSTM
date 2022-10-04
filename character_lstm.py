# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 14:01:16 2021

@author: narut
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CharacterLSTM(nn.Module):
  def __init__(self, vocab_size, embedding_dim=128, hidden_dim=512, 
               dense_dim=256, n_layers=1, device = None):
    
    super().__init__()  
    self.n_layers = n_layers
    self.hidden_dim = hidden_dim
    self.device = device
    self.embedding = nn.Embedding(vocab_size, embedding_dim)
    self.LSTM = nn.LSTM(embedding_dim, hidden_dim, num_layers = n_layers, batch_first = True, dropout = 0.3)
    self.fc1 = nn.Linear(hidden_dim, vocab_size)
    #self.fc2 = nn.Linear(dense_dim, vocab_size)
    
    """
    self.embedding = nn.Embedding(vocab_size, embedding_dim, max_norm=max_norm)
    self.LSTM = nn.LSTM(embedding_dim, hidden_dim, num_layers = n_layers, batch_first = True, dropout = 0.3)
    self.fc1 = nn.Linear(hidden_dim, dense_dim)
    self.fc2 = nn.Linear(dense_dim, vocab_size)
    """
  def forward(self, x):
    batch_size = x.shape[0]
    h0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).requires_grad_().to(self.device)
    c0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).requires_grad_().to(self.device)
    
    embeds = self.embedding(x)
    lstm_out, (h, c) = self.LSTM(embeds, (h0, c0))  
    #lstm_out, (h, c) = self.LSTM(embeds)  
    h = h[-1, :, :].squeeze()
    fc1_out = self.fc1(h)
    #fc2_out = self.fc2(fc1_out)
    #char_score = F.log_softmax(fc1_out, dim=1)
    
    return fc1_out
  