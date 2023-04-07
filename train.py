import os
import time
import warnings

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.optim.adam import Adam
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


data = pd.read_csv('../Data/서울_data.csv')

df = Processing(data)
x, y, x_ss, y_ms, train_loader, test_loader = split_data(df, 600, 60, 7)

input_size = x_ss.size(2)
num_layers = 1
hidden_size = 16
sequence_length = 60

model = GRU(input_size = input_size,
            hidden_size = hidden_size,
            sequence_length = sequence_length,
            num_layers = num_layers, 
            dropout = 0.3, 
            device = device).to(device)

criterion = nn.MSELoss()
lr = 1e-3
num_epochs = 10000
optimizer = Adam(model.parameters(), lr = lr)
patience = 10

loss_list = []
n = len(train_loader)

for epoch in range(num_epochs):
    running_loss = 0.0
    
    for data in train_loader:
        seq, target = data
        out = model(seq)
        loss = criterion(out, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    loss_list.append(running_loss/n)
    if (epoch+1) % 100 == 0:
        print('epoch: %d loss: %.4f'%(epoch+1, running_loss/n))
        
    if (epoch % patience == 0) & (epoch != 0):
            
            if loss_list[epoch-patience] < loss_list[epoch]:
                print('\n Early Stopping / epoch: %d loss: %.4f'%(epoch+1, running_loss/n))
                
                break
   

PATH = f'model/{model}.pth'
Save_model(model, PATH)