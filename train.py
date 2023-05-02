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

from utils import *
from model import *

#warnings.filterwarnings('ignore')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Train:
    def __init__(self):
    
    
    def Trainer(self, model, train_loader, test_laoder):
    
        input_size = x_ss.size(2)
        num_layers = 1
        hidden_size = 16
        sequence_length = 60

        lr = 1e-3
        num_epochs = 10000
        optimizer = Adam(model.parameters(), lr = lr)
        criterion = nn.MSELoss()
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
        
        learned_model = model.state_dict()
        
        return learned_model
        
    def Train(self):
    
        path = 'Data/CleanedData/*.csv'
        file_list, name_list = Load_files.load_files(path)
        
        df = Prepare_df.Processing(file)
        x, y, x_ss, y_ms, train_loader, test_loader = Prepare_df.Split_data(df, int(len(df)*0.75), 60, 7)
        
        models = []
        
        learned_model = Trainer(model, train_loader, test_loader)
        
        path_model = f'model/{model}.pth'
        Save_and_Load.save_and_load(learned_model, path_model)

