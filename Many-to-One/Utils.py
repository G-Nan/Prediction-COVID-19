import torch
import pandas as pd

def load_data():
    
    data = pd.read_csv()
    
    x = data.iloc[:, 1:]
    y = data.iloc[:, :1]
    
    x = torch.from_numpy(x.values)
    y = torch.from_numpy(y.values)
    
    return x, y
    
def split_data(x, y):

    train_cnt = 
    valid_cnt = 
    
    x = x.split([train_cnt, valid_cnt], dim = 0)
    y = y.split([train_cnt, valid_cnt], dim = 0)

    return x, y

def get_hidden_sizes(input_size, output_size, n_layers):
    step_size = int((input_size - output_size) / n_layers)
    
    hidden_sizes = []
    current_size = input_size
    for i in range(n_layers - 1):
        hidden_size += [current_size - step_size]
        current_size = hidden_sizes[-1]
        
    return hidden_sizes

class EarlyStopping:
    def __init__(self, patience=5):
        self.loss = np.inf
        self.patience = 0
        self.patience_limit = patience
        
    def step(self, loss):
        if self.loss > loss:
            self.loss = loss
            self.patience = 0
        else:
            self.patience += 1
    
    def is_stop(self):
        return self.patience >= self.patience_limit