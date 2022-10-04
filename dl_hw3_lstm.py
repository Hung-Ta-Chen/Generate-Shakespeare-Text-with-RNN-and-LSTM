# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 17:00:01 2021

@author: narut
"""

import numpy as np
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from character_dataset import CharacterDataset
from character_lstm import CharacterLSTM

def train(net, train_loader, optimizer, criterion, epoch_idx, device=None):
  net.train()
  running_loss = 0.0
  batch_cnt = 0
  for batch_idx, (inputs, labels) in enumerate(train_loader):
    if device != None:
      inputs, labels = inputs.to(device), labels.to(device)
  
    optimizer.zero_grad()   # zero the parameter gradients
    outputs = net(inputs)   # Forward
    loss = criterion(outputs, labels)
    loss.backward()   # Backprop
    optimizer.step()  # Update parameters

    running_loss += loss.item()
    batch_cnt = batch_idx

    if batch_idx % 200 == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(  
        epoch_idx, batch_idx * len(inputs), len(train_loader.dataset),      
        100. * batch_idx / len(train_loader), loss.item()))
      
  return (running_loss / batch_cnt)


def test(net, test_loader, criterion, device=None):
  net.eval()
  test_loss = 0.0
  correct = 0.0
  batch_cnt = 0
  with torch.no_grad():
    for batch_idx, (inputs, labels) in enumerate(test_loader):
      if device != None:
        inputs, labels = inputs.to(device), labels.to(device)
      outputs = net(inputs)
      c = criterion(outputs, labels)
      test_loss += c.item()  # sum up batch loss
      pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
      correct += pred.eq(labels.view_as(pred)).sum().item()
      batch_cnt = batch_idx
      
  test_loss /= batch_cnt

  print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
  
  return 100. - (100. * correct / len(test_loader.dataset))

    
def generate_text(net, dataset, init_str, text_len, device = None):
  result = init_str
  net.eval()
  
  for i in range(text_len):
    if result == init_str:
      prev_char = init_str
    else:
      if len(result) < 100:
        prev_char = result[-len(result):-1]
      else:
        prev_char = result[-100:-1]

    encode = torch.LongTensor([[dataset.char2idx[c] for c in prev_char]]).to(device)
    
    logit = net(encode)
    probs = F.softmax(logit, dim=0).detach().cpu().numpy()
    #print(dataset.char_set)
    #print(probs)
    new_char = np.random.choice(dataset.char_set, p=probs)
    result += new_char
  
  return result
    
  
  
  
def main():
  # ----------------------------------
  # Pre-processing data
  # ----------------------------------
  data_URL = 'shakespeare_train.txt'
  data_val_URL = 'shakespeare_valid.txt'
  
  with io.open(data_URL, 'r', encoding='utf8') as f:
    train_text = f.read()
    
  with io.open(data_val_URL, 'r', encoding='utf8') as f:
    val_text = f.read()
  
  # Character's collection
  char_set = set(train_text)
  vocab_size = len(char_set)
  """
  # Construct character's dictionary
  char2int = {c:i for i ,c in enumerate(chars)}
  int2char = dict(enumerate(chars))
  
  # Encode data, shape = [number of characters]
  train_data = np.array([char2int[c] for c in text] , dtype = np.int32)
  valid_data = np.array([char2int[c] for c in text_val] , dtype = np.int32)
  """
  
  # training parameter
  batch_size = 128
  epoch_num = 10
  seq_len = 100
  
  train_dataset = CharacterDataset(train_text, seq_len = seq_len)
  val_dataset = CharacterDataset(val_text, seq_len = seq_len, char_set=char_set)
  train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
  train_loader_val = DataLoader(train_dataset, batch_size = batch_size, shuffle = False)
  val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)
  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(device)
  net = CharacterLSTM(len(char_set), device = device, hidden_dim = 768).to(device)
  
  #optimizer = optim.Adam(net.parameters(), lr = 1e-2)
  #optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)
  optimizer = optim.Adam(net.parameters(), lr = 2e-4)
  criterion = nn.CrossEntropyLoss()
  
  
  # Training
  loss_list = []
  train_err_list = []
  test_err_list = []
  for epoch in range(1, epoch_num + 1):
    loss = train(net, train_loader, optimizer, criterion, epoch, device)
    loss_list.append(loss)

    train_err = test(net, train_loader_val, criterion, device)
    train_err_list.append(train_err)
    test_err = test(net, val_loader, criterion, device)
    test_err_list.append(test_err)
    torch.save(net.state_dict(), 'lstm_net_{}.pt'.format(epoch))

  # Save parameters of the model
  torch.save(net.state_dict(), 'lstm_net_full.pt')
  
  """
  # Load the parameters of model
  net.load_state_dict(torch.load('lstm_net_full_768.pt'))

  """
  # Plot Accuracy
  
  print("=== Show error rate plot ===>>")
  fig , ax = plt.subplots()
  #plt.rcParams["figure.figsize"] = (8, 3.5)
  plt.plot(range(len(train_err_list)), train_err_list, label = "training error rate", color = "blue", linewidth = 0.5)
  plt.plot(range(len(test_err_list)), test_err_list, label = "testing error rate", color = "orange", linewidth = 0.5)
  plt.title("Error Rate of the Model")
  plt.ylabel("Error Rate(%)")
  plt.xlabel("Epoch")
  leg = ax.legend(loc='lower right') 
  plt.savefig('lstm_err_rate.png')
  plt.show()
  
  print(" ")
  
  # Plot learning curve
  print("=== Show learning plot ===>>")
  plt.plot(loss_list)
  plt.ylabel("Loss")
  plt.xlabel("Epoch")
  plt.title("Learning Curve")
  plt.savefig('lstm_lc.png')
  plt.show()
  
  print(" ")
  """
  
  result = generate_text(net, train_dataset, "KING LEAR:\n", 1500, device = device)
  print(result)
  """
  
if __name__=="__main__":
  main()
  