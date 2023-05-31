import os
import time
import warnings

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from tqdm.notebook import tqdm
from torch.optim.adam import Adam
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from utils import *
from model import *

#warnings.filterwarnings('ignore')

class Trainer:

    def Many_to_One(train_loader, test_loader, model, criterion, optimizer, num_epochs, patience, device):
    
        n = len(train_loader)
        loss_list = []
        min_loss = np.inf
        
        epoch_iterator = tqdm(range(num_epochs), position = 1)
        for epoch in epoch_iterator:
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
            
            #if (epoch+1) % 100 == 0:
            #    print('epoch: %d loss: %.4f'%(epoch+1, running_loss/n))
        
            if (running_loss/n) <= min_loss:
                min_loss = (running_loss/n)
                k = 0
            else:
                k += 1
            
            epoch_iterator.set_description("Training (%d / %d ) (loss=%2.5f)" % (epoch+1, num_epochs, running_loss/n))
            
            if k > patience:
                print('\n Early Stopping / epoch: %d loss: %.4f'%(epoch+1, running_loss/n))
                break
            
            
        return loss_list, model, epoch
        
    def Many_to_Many(train_loader, test_loader, model, criterion, optimizer, num_epochs, patience, device):
    
        n = len(train_loader)
        loss_list = []
        min_loss = np.inf
        
        epoch_iterator = tqdm(range(num_epochs), position = 1)
        for epoch in epoch_iterator:
            running_loss = 0.0
    
            for data in train_loader:
                seq, target = data
                out = model(seq, target, 7, 0.5, device).to(device)
                loss = criterion(out, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            loss_list.append(running_loss/n)
            #if (epoch+1) % 100 == 0:
            #    print('epoch: %d loss: %.6f'%(epoch+1, running_loss/n))
        
        
            if (running_loss/n) <= min_loss:
                min_loss = (running_loss/n)
                k = 0
            else:
                k += 1
            
            epoch_iterator.set_description("Training (%d / %d ) (loss=%2.5f)" % (epoch+1, num_epochs, running_loss/n))
            
            if loss_list[epoch+1-patience] < loss_list[epoch+1]:
                print('\n Early Stopping / epoch: %d loss: %.6f'%(epoch+1, running_loss/n))
                break
                    
        return loss_list, model, epoch