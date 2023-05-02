import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#warnings.filterwarnings('ignore')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Load_files:

    def __init__(self, path):
        self.path = path
    
    def load_files(self, path):
    
        names = glob.glob(path)
        file_list = []
        name_list = []
        for i, name in enumerate(names):
            assert len(name) == 31
            name_list.append([name[17:19]])
            sub = pd.read_csv(name)
            file_list.append(sub)
        
        return file_list, name_list

class Prepare_df:

    def __init__(self, file):
        self.file = file
        self.df = df
        
    def processing(self, data):

        df = data.loc[:, ['stdDay', 'defCnt']]
        df.rename(columns = {'stdDay':'Date', 'defCnt':'ACC'}, inplace = True)
        df['AC'] = df['ACC'] - df['ACC'].shift(1)
        df['DAC'] = df['AC'] - df['AC'].shift(1)
        df['DDAC'] = df['DAC'] - df['DAC'].shift(1)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace = True)
        df = df.loc[:, ['AC', 'DAC', 'DDAC']]
        df = df.dropna(axis = 0)
    
        return df

    def scailing(self, x, y):

        ms = MinMaxScaler()
        ss = StandardScaler()

        x_ss = ss.fit_transform(x)
        y_ms = ms.fit_transform(y)

        return x_ss, y_ms
        
    def window_sliding(self, df, x, y, iw, ow):
    
        x_ws, y_ws = list(), list()
        for i in range(len(df)):
            x_end = i + iw
            y_end = x_end + ow
        
            if y_end > len(df):
                break
        
            tx = x[i:x_end, :]
            ty = y[x_end:y_end, :]
        
            x_ws.append(tx)
            y_ws.append(ty)
    
        return torch.FloatTensor(np.array(x_ws)).to(device), torch.FloatTensor(np.array(y_ws)).to(device)
    
    def split_data(self, df, train_len, input_window, output_window):
        x = df.iloc[:, 0:]
        y = df.iloc[:,:1]
    
        x_ss, y_ms = Prepare_df.Scailing(x, y)
       
        x = x.to_numpy()
        y = y.to_numpy()
        x, y = Prepare_df.Window_sliding(df, x, y, input_window, output_window)
        x_ss, y_ms = Prepare_df.Window_sliding(df, x_ss, y_ms, input_window, output_window)

        x_train = x_ss[:train_len]
        y_train = y_ms[:train_len]
        x_test = x_ss[train_len:]
        y_test = y_ms[train_len:]

        train = torch.utils.data.TensorDataset(x_train, y_train)
        test = torch.utils.data.TensorDataset(x_test, y_test)

        batch_size = 64
        train_loader = torch.utils.data.DataLoader(dataset = train, batch_size = batch_size, shuffle = False)
        test_loader = torch.utils.data.DataLoader(dataset = test, batch_size = batch_size, shuffle = False)

        return x, y, x_ss, y_ms, train_loader, test_loader
        
    
class Save_and_Load:

    def __init__(self, model, path_model):
        self.model = model
        self.path_model = path_model
    
    def save_model(model, path_model):
        torch.save(model.state_dict(), path_model)

    def load_model(model, path_model):
        model.load_state_dict(torch.load(path_model), strict=False)
        model.eval()
        
    def save_and_load(model, path_model):
        save_model(model, path_model)
        load_model(model, path_model)
        
def plotting(train_loader, test_loader, actual):
    with torch.no_grad():
        train_pred = []
        test_pred = []

        for data in train_loader:
            seq, target = data
            out = model(seq)
            train_pred += out.cpu().numpy().tolist()

        for data in test_loader:
            seq, target = data
            out = model(seq)
            test_pred += out.cpu().numpy().tolist()
      
    total = train_pred + test_pred
    plt.figure(figsize=(20,10))
    plt.plot(np.ones(100)*len(train_pred), np.linspace(0,1,100), '--', linewidth=0.6)
    plt.plot(actual, '--')
    plt.plot(total, 'b', linewidth=0.6)

    plt.legend(['train boundary', 'actual', 'prediction'])
    plt.show()

def mae(true, pred):
    return np.mean(np.abs(true-pred))
    
def rmse(true, pred):
    return np.mean((true-pred)**2)**(1/2)
    
def mape(true, pred):
    return 100 * np.mean(np.abs((true-pred)/true))

def predict():
    pre7 = ms.inverse_transform(predicted.reshape(881, 7))
    lab7 = ms.inverse_transform(label_y.reshape(881, 7))

    predicted_final = np.vstack((first_predicted[:ran], pre7[ran].reshape(7, 1)))
    label_y_final = np.vstack((first_label_y[:ran], lab7[ran].reshape(7, 1)))

    plt.figure(figsize = (10, 6)) 
    plt.axvline(x = 42, c = 'g', linestyle = '--')

    plt.plot(label_y_final[-50:], label = 'Actual Data')
    plt.plot(predicted_final[-50:], label = 'Predicted Data')
    plt.title('Time-Series Prediction')
    plt.legend()
    plt.show()

def random_predict():
    ran = random.randrange(600, 899)
    plt.plot(lab7[ran], label = 'Actual Data')
    plt.plot(pre7[ran], label = 'Predicted Data')
    plt.show()