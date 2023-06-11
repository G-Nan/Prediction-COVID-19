import os
import glob
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from model import *

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
            sub = pd.read_csv(name, encoding = 'cp949')
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
        
        train = torch.utils.data.TensorDataset(x_train, y_train)
        test = torch.utils.data.TensorDataset(x_test, y_test)

        batch_size = batch_size
        train_loader = torch.utils.data.DataLoader(dataset = train, batch_size = batch_size, shuffle = False)
        test_loader = torch.utils.data.DataLoader(dataset = test, batch_size = batch_size, shuffle = False)

        return x, y, x_ss, y_ms, train_loader, test_loader
   
def save_model(model_dict, path_model):
    torch.save(model_dict, path_model)

def load_model(model, path_model):
    model.load_state_dict(torch.load(path_model), strict=False)
    model.eval()
    return model
        
def save_and_load_model(model, path_model):
    save_model(model, path_model)
    load_model(model, path_model)
    
def load_model_multiple(dic_hyperparameter, var1, var2):
    input_size = 3
    sequence_length = 60
    dic_loaded_model = {}
    
    models_list = ['RNN', 'LSTM', 'GRU', 'BiRNN', 'BiLSTM', 'BiGRU', 
               'seq2seq_RNN', 'seq2seq_LSTM', 'seq2seq_GRU', 
               'seq2seq_BiRNN', 'seq2seq_BiLSTM', 'seq2seq_BiGRU']
               
    for num_model in range(12):
    
        model_name = models_list[num_model]
        
        lr = dic_hyperparameter[model_name][1]
        patience = dic_hyperparameter[model_name][2]
        num_layers = dic_hyperparameter[model_name][3]
        batch_size = dic_hyperparameter[model_name][4]
        hidden_size = dic_hyperparameter[model_name][5]
        dropout = dic_hyperparameter[model_name][6]
        if len(dic_hyperparameter[model_name]) < 8:
            criterion = nn.MSELoss()
        else:
            criterion = dic_hyperparameter[model_name][7]
            
            
        if num_model == 0:
            model = RNN(input_size = input_size,
                        hidden_size = hidden_size,
                        sequence_length = sequence_length,
                        num_layers = num_layers, 
                        dropout = dropout, 
                        device = device).to(device)

        elif num_model == 1:
            model = LSTM(input_size = input_size,
                         hidden_size = hidden_size,
                         sequence_length = sequence_length,
                         num_layers = num_layers, 
                         dropout = dropout, 
                         device = device).to(device)

        elif num_model == 2:
            model = GRU(input_size = input_size,
                        hidden_size = hidden_size,
                        sequence_length = sequence_length,
                        num_layers = num_layers, 
                        dropout = dropout, 
                        device = device).to(device)

        elif num_model == 3:
            model = BiRNN(input_size = input_size,
                          hidden_size = hidden_size,
                          sequence_length = sequence_length,
                          num_layers = num_layers, 
                          dropout = dropout, 
                          device = device).to(device)

        elif num_model == 4:
            model = BiLSTM(input_size = input_size,
                           hidden_size = hidden_size,
                           sequence_length = sequence_length,
                           num_layers = num_layers, 
                           dropout = dropout, 
                           device = device).to(device)

        elif num_model == 5:
            model = BiGRU(input_size = input_size,
                          hidden_size = hidden_size,
                          sequence_length = sequence_length,
                          num_layers = num_layers, 
                          dropout = dropout, 
                          device = device).to(device)
                
        elif num_model == 6:
            model = RNN_encoder_decoder(input_size = input_size, 
                                        hidden_size = hidden_size,
                                        num_layers = num_layers, 
                                        dropout = dropout,
                                        device = device).to(device)

        elif num_model == 7:
            model = LSTM_encoder_decoder(input_size = input_size, 
                                         hidden_size = hidden_size,
                                         num_layers = num_layers, 
                                         dropout = dropout,
                                         device = device).to(device)

        elif num_model == 8:
            model = GRU_encoder_decoder(input_size = input_size, 
                                        hidden_size = hidden_size,
                                        num_layers = num_layers, 
                                        dropout = dropout,
                                        device = device).to(device)

        elif num_model == 9:
            model = BiRNN_encoder_decoder(input_size = input_size, 
                                            hidden_size = hidden_size,
                                            num_layers = num_layers, 
                                            dropout = dropout,
                                            device = device).to(device)

        elif num_model == 10:
            model = BiLSTM_encoder_decoder(input_size = input_size, 
                                              hidden_size = hidden_size,
                                              num_layers = num_layers, 
                                              dropout = dropout,
                                              device = device).to(device)

        elif num_model == 11:
            model = BiGRU_encoder_decoder(input_size = input_size, 
                                            hidden_size = hidden_size,
                                            num_layers = num_layers, 
                                            dropout = dropout,
                                            device = device).to(device)
            
        dic_loaded_model[model_name] = load_model(model, f'model/{var1}/{var2}/{model_name}.pth')
    
    return dic_loaded_model
    
def save_hyperparameter(hyperparameter, path_hyperparameter):
    with open(path_hyperparameter, 'wb') as f:
        pickle.dump(hyperparameter, f)
    
def load_hyperparameter(path_hyperparameter):
    with open(path_hyperparameter, 'rb') as f:
        hyperparameter = pickle.load(f)
    return hyperparameter

def save_and_load_hyperparameter(hyperparameter, path_hyperparameter):
    save_hyperparameter(hyperparameter, path_hyperparameter)
    load_hyperparameter(hyperparameter, path_hyperparameter)
    
def plotting(label_y, predicted, bar):
    
    plt.figure(figsize = (10, 6))
    plt.axvline(x = bar, c = 'r', linestyle = '--')

    plt.plot(label_y, label = 'Actual Data')
    plt.plot(predicted, label = 'Predicted Data')
    plt.title('Time-Series Prediction')
    plt.legend()
    plt.show()

def criterion2(actual, predict):
    
    loss = 0
    div = 0
    
    for i in range(7):
        div += (i+1)
    
    for i in range(7):
        loss += sum((i+1) * (abs(actual[:, i, 0] - predict[:, i, 0])))

    loss /= div
    loss /= actual.shape[0]
        
    return loss

def criterion3(actual, predict):
    
    loss = 0
    div = 0
    
    for i in range(7):
        div += (i+1)**2
    
    for i in range(7):
        loss += sum(((i+1)**2) * ((actual[:, i, 0] - predict[:, i, 0])**2))
               
    loss /= div   
    loss = loss**(1/2) 
    loss /= actual.shape[0]
    
    return loss

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
    
def predict_mtm(model, df, x_ss, y_ms, target_len, teacher_forcing_ratio, device):

    x = df.iloc[:, 0:]
    y = df.iloc[:,:1]

    ms = MinMaxScaler()
    ss = StandardScaler()

    ss.fit(x)
    ms.fit(y)

    train_predict = model(x_ss, y_ms, target_len, 0.5, device)
    predicted = train_predict.cpu().detach().numpy()
    label_y = y_ms.cpu().detach().numpy()

    predicted = predicted.reshape(-1, 1)
    label_y = label_y.reshape(-1, 1)
    
    predicted = ms.inverse_transform(predicted)
    label_y = ms.inverse_transform(label_y)
    
    predicted = predicted.reshape(-1, target_len, 1)
    label_y = label_y.reshape(-1, target_len, 1)
    
    first_predicted = predicted[:, 0, 0].reshape(-1, 1)
    first_label_y = label_y[:, 0, :].reshape(-1, 1)
    
    all_predicted = np.concatenate((first_predicted[:-1], predicted[-1]))

    return label_y, predicted, first_label_y, first_predicted, all_predicted

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
    
    
    
def N_in_dict(dic_files, list_N):
    
    i = 0
    for key, file in dic_files.items():
        dic_files[key] = [list_N[i], file]
        i += 1
    
    return dic_files

def processing_SIR(file, N, Recover):

    df = pd.DataFrame()

    df['Date'] = file['stdDay']
    df['City'] = file['gubun']

    # 사망자
    df['Dead'] = file['deathCnt']                 # 누적 사망자
    df['daily_Dead'] = file['deathCnt'].diff()    # 일일 사망자

    # 감염자
    df['Inf_AC'] = file['defCnt']                 # 누적 감염자
    df['Infected'] = file['defCnt']  
    df.iloc[Recover:, -1] = (df.iloc[Recover:, -1]
                            .reset_index(drop = True)
                            .sub(df.iloc[:len(df)-Recover, -1]))
    
    # 회복자
    df['Recovered'] = 0
    df.iloc[Recover:, -1] = df.iloc[:len(df)-Recover, 4]
    df['Recovered'] = df['Recovered'] - df['Dead']

    # 취약자
    df['Susceptible'] = N - df['Infected'] - df['Dead'] - df['Recovered']
    df = df[['Date', 'City', 'Susceptible', 'Infected', 'Dead', 'Recovered']]
    
    # Alpha
    df['alpha'] = (df['Susceptible'].shift(-1) - df['Susceptible'])
    df['alpha'] = (-1 * N * df['alpha'])/(df['Susceptible'] * df['Infected'])
    
    # Beta
    df['beta'] = (df['Recovered'].shift(-1) - df['Recovered'])
    df['beta'] = df['beta']/df['Infected']

    # Gamma
    df['gamma'] = (df['Dead'].shift(-1) - df['Dead'])
    df['gamma'] = df['gamma']/df['Infected']

    df.loc[df['alpha'] == 0, 'alpha'] = 0
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return df

def multiple_processing_SIR(dic_SIRs, df_variants, city, file, variable1, variable2, 
                            input_index1, output_index1, input_index2, output_index2, 
                            Recover, target_len):
    
    var1 = df_variants[variable1][input_index1:output_index1].reset_index(drop = True)
    var2 = df_variants[variable2][input_index1:output_index1].reset_index(drop = True)


    file['dailyDeath'] = file['deathCnt'].diff().fillna(0)
    df = file.iloc[input_index2:output_index2, [4, 10, 11]].reset_index(drop = True).copy()

    df[f'{variable1}_daily_dead'] = 0
    df.iloc[Recover:, -1] = var1[:-Recover]*df['dailyDeath'][Recover:].reset_index(drop = True)
    df[f'{variable2}_daily_dead'] = 0
    df.iloc[Recover:, -1] = var2[:-Recover]*df['dailyDeath'][Recover:].reset_index(drop = True)

    df[f'{variable1}_Dead'] = 0
    df.iloc[Recover:, -1] = var1[:-Recover]*df['dailyDeath'][Recover:].reset_index(drop = True)
    df[f'{variable2}_Dead'] = 0
    df.iloc[Recover:, -1] = var2[:-Recover]*df['dailyDeath'][Recover:].reset_index(drop = True)

    df[df.columns[5:7]] = df[df.columns[5:7]].expanding().sum() 

    df[f'{variable1}_daily_Inf'] = df['incDec']*var1
    df[f'{variable2}_daily_Inf'] = df['incDec']*var2

    df[f'{variable1}_Inf_AC'] = df['incDec']*var1
    df[f'{variable2}_Inf_AC'] = df['incDec']*var2

    df[f'{variable1}_Infected'] = df['incDec']*var1
    df[f'{variable2}_Infected'] = df['incDec']*var2

    df.iloc[:, 9:] = df.iloc[:, 9:].expanding().sum() 

    df.iloc[Recover:, -2] = (df.iloc[Recover:, -2]
                             .reset_index(drop = True)
                             .sub(df.iloc[:len(df)-Recover, -2]))
    df.iloc[Recover:, -1] = (df.iloc[Recover:, -1]
                             .reset_index(drop = True)
                             .sub(df.iloc[:len(df)-Recover, -1]))

    df[f'{variable1}_Recovered'] = 0
    df.iloc[Recover:, -1] = df.iloc[:len(df)-Recover, 9]
    df[f'{variable2}_Recovered'] = 0
    df.iloc[Recover:, -1] = df.iloc[:len(df)-Recover, 10]

    df[f'{variable1}_Recovered'] = df[f'{variable1}_Recovered'] - df[f'{variable1}_Dead']
    df[f'{variable2}_Recovered'] = df[f'{variable2}_Recovered'] - df[f'{variable2}_Dead']

    df['Susceptible'] = dic_SIRs[city]['Susceptible'][input_index2:output_index2].reset_index(drop = True)


    df[f'{variable1}_alpha'] = df[f'{variable1}_daily_Inf'].shift(-1)
    df[f'{variable1}_alpha'] = ((dic_SIRs[city]['Susceptible'][0] * df[f'{variable1}_alpha'])
                               / (df['Susceptible'] * df[f'{variable1}_Infected']))

    df[f'{variable2}_alpha'] = df[f'{variable2}_daily_Inf'].shift(-1)
    df[f'{variable2}_alpha'] = ((dic_SIRs[city]['Susceptible'][0] * df[f'{variable2}_alpha'])
                               / (df['Susceptible'] * df[f'{variable2}_Infected']))

    df[f'{variable1}_beta'] = (df[f'{variable1}_Recovered'].shift(-1) - df[f'{variable1}_Recovered']) / df[f'{variable1}_Infected']
    df[f'{variable2}_beta'] = (df[f'{variable2}_Recovered'].shift(-1) - df[f'{variable2}_Recovered']) / df[f'{variable2}_Infected']

    df[f'{variable1}_gamma'] = (df[f'{variable1}_Dead'].shift(-1) - df[f'{variable1}_Dead']) / df[f'{variable1}_Infected']
    df[f'{variable2}_gamma'] = (df[f'{variable2}_Dead'].shift(-1) - df[f'{variable2}_Dead']) / df[f'{variable2}_Infected']    

    df = df[['stdDay', 'Susceptible', 
            f'{variable1}_Infected', f'{variable1}_Recovered', f'{variable1}_Dead', 
            f'{variable1}_alpha', f'{variable1}_beta', f'{variable1}_gamma',
            f'{variable2}_Infected', f'{variable2}_Recovered', f'{variable2}_Dead', 
            f'{variable2}_alpha', f'{variable2}_beta', f'{variable2}_gamma']]

    df.loc[df[f'{variable1}_alpha'] == 0, f'{variable1}_alpha'] = 0
    df.loc[df[f'{variable2}_alpha'] == 0, f'{variable2}_alpha'] = 0
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)[target_len:].reset_index(drop = True)
        
    df
    
    return df