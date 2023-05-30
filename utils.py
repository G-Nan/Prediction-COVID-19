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

    def __init__(self, path, name_start, name_end):
        self.path = path
    
    def load_files(path, name_start, name_end):
    
        names = glob.glob(path)
        dic_files = {}
        for name in names:
            city = name[name_start:name_end]
            sub = pd.read_csv(name)
            dic_files[city] = sub

        return dic_files

class Prepare_df:
        
    def processing(data, variable1, variable2):
    
        df = data.loc[:, [variable1, variable2]]
        df.rename(columns = {variable1:'Date', variable2:'AC'}, inplace = True)
        df['DAC'] = df['AC'] - df['AC'].shift(1)
        df['DDAC'] = df['DAC'] - df['DAC'].shift(1)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace = True)
        df = df.loc[:, ['AC', 'DAC', 'DDAC']]
        df = df.dropna(axis = 0)
    
        return df

    def scailing(x, y):
        
        ms = MinMaxScaler()
        ss = StandardScaler()

        x_ss = ss.fit_transform(x)
        y_ms = ms.fit_transform(y)

        return x_ss, y_ms
        
    def window_sliding(df, x, y, iw, ow, method):
    
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
    
        if method == 'mto':
            x_ss = torch.FloatTensor(np.array(x_ws)).to(device)
            y_ms = torch.FloatTensor(np.array(y_ws)).to(device).view([-1, 1])
        if method == 'mtm':
            x_ss = torch.FloatTensor(np.array(x_ws)).to(device)
            y_ms = torch.FloatTensor(np.array(y_ws)).to(device)
        return x_ss, y_ms
    
    def split_data(df, train_len, input_window, output_window, batch_size, method):
    
        x = df.iloc[:, 0:]
        y = df.iloc[:,:1]
    
        x_ss, y_ms = Prepare_df.scailing(x, y)
       
        x = x.to_numpy()
        y = y.to_numpy()
        x, y = Prepare_df.window_sliding(df, x, y, input_window, output_window, method)
        x_ss, y_ms = Prepare_df.window_sliding(df, x_ss, y_ms, input_window, output_window, method)

        x_train = x_ss[:train_len]
        y_train = y_ms[:train_len]
        x_test = x_ss[train_len:]
        y_test = y_ms[train_len:]

        print(x_train.shape)
        print(y_train.shape)
        print(x_test.shape)
        print(y_test.shape)

        train = torch.utils.data.TensorDataset(x_train, y_train)
        test = torch.utils.data.TensorDataset(x_test, y_test)

        batch_size = batch_size
        train_loader = torch.utils.data.DataLoader(dataset = train, batch_size = batch_size, shuffle = False)
        test_loader = torch.utils.data.DataLoader(dataset = test, batch_size = batch_size, shuffle = False)

        return x, y, x_ss, y_ms, train_loader, test_loader

    
def save_model(model, path_model):
    torch.save(model.state_dict(), path_model)

def load_model(model, path_model):
    model.load_state_dict(torch.load(path_model), strict=False)
    model.eval()
        
def save_and_load(model, path_model):
    save_model(model, path_model)
    load_model(model, path_model)
        
def plotting(label_y, predicted, bar):
    
    plt.figure(figsize = (10, 6))
    plt.axvline(x = bar, c = 'r', linestyle = '--')

    plt.plot(label_y, label = 'Actual Data')
    plt.plot(predicted, label = 'Predicted Data')
    plt.title('Time-Series Prediction')
    plt.legend()
    plt.show()

def mae(true, pred):
    return np.mean(np.abs(true-pred))
    
def rmse(true, pred):
    return np.mean((true-pred)**2)**(1/2)
    
def mape(true, pred):
    return 100 * np.mean(np.abs((true-pred)/true))

#class Predict:
def predict_mto(model, df, x_ss, y_ms):

    x = df.iloc[:, 0:]
    y = df.iloc[:,:1]

    ms = MinMaxScaler()
    ss = StandardScaler()

    ss.fit(x)
    ms.fit(y)

    train_predict = model(x_ss)
    predicted = train_predict.cpu().data.numpy()
    label_y = y_ms.cpu().data.numpy()
    
    #predicted = predicted.reshape(1110, 1)
    predicted = ms.inverse_transform(predicted)
    label_y = ms.inverse_transform(label_y)

    return label_y, predicted
    
def predict_mtm(model, df, x_ss, y_ms, len_df, target_len, teacher_forcing_ratio, device):

    x = df.iloc[:, 0:]
    y = df.iloc[:,:1]

    ms = MinMaxScaler()
    ss = StandardScaler()

    ss.fit(x)
    ms.fit(y)

    train_predict = model(x_ss, y_ms, 7, 0.5, device)
    predicted = train_predict.cpu().data.numpy()
    label_y = y_ms.cpu().data.numpy()
    
    first_predicted = predicted[:, 0, 0].reshape(len_df, 1)
    first_label_y = label_y[:, 0, :].reshape(len_df, 1)

    first_predicted = ms.inverse_transform(first_predicted)
    first_label_y = ms.inverse_transform(first_label_y)

    return label_y, predicted, first_label_y, first_predicted

def predict_point(label_y, predicted):

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

def random_predict(lab7, pre7):

    ran = random.randrange(600, 899)
    plt.plot(lab7[ran], label = 'Actual Data')
    plt.plot(pre7[ran], label = 'Predicted Data')
    plt.show()