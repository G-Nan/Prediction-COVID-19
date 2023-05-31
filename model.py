import torch
import torch.nn as nn
import random

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, sequence_length, num_layers, dropout, device):
        super(RNN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size,
                          hidden_size, 
                          num_layers, 
                          batch_first = True, 
                          bidirectional = False, 
                          dropout = dropout)
        self.fc1 = nn.Linear(hidden_size * sequence_length, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):    
        # x_size = [batch_size, sequence_length, features]
        h0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size).to(self.device)
        out, hn = self.rnn(x, h0)
        # out_size = [batch_size, sequence_length, features]
        # hn_size = [num_layers, batch_size, hidden_size]
        out = out.reshape(out.shape[0], -1)
        # out_size = [batch_size, sequence_length * features]
        out = self.fc1(out)
        # out_size = [batch_size, 128]
        out = self.relu(out)
        out = self.fc2(out)
        # out_size = [batch_size, 64]
        out = self.relu(out)
        out = self.fc3(out)
        # out_size = [batch_size, 1]
        return out    

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, sequence_length, num_layers, dropout, device):
        super(LSTM, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, 
                            hidden_size, num_layers, 
                            batch_first = True, 
                            bidirectional = False, 
                            dropout = dropout)
        self.fc1 = nn.Linear(hidden_size * sequence_length, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x_size = [batch_size, sequence_length, features]
        h0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size).to(self.device)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        # out_size = [batch_size, sequence_length, hidden_size]
        # hn_size = [num_layers, batch_size, hidden_size]
        # cn_size = [num_layers, batch_size, hidden_size]
        out = out.reshape(out.shape[0], -1)
        # out_size = [batch_size, sequence_length * hidden_size]
        out = self.fc1(out)
        # out_size = [batch_size, 128]
        out = self.relu(out)
        out = self.fc2(out)
        # out_size = [batch_size, 64]
        out = self.relu(out)
        out = self.fc3(out)
        # out_size = [batch_size, 1]
        return out

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, sequence_length, num_layers, dropout, device):
        super(GRU, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, 
                          hidden_size, 
                          num_layers, 
                          batch_first = True, 
                          bidirectional = False, 
                          dropout = dropout)
        self.fc1 = nn.Linear(hidden_size * sequence_length, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x_size = [batch_size, sequence_length, features]
        h0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size).to(self.device)
        out, hn = self.gru(x, h0)
        # out_size = [batch_size, sequence_length, hidden_size]
        # hn_size = [num_layers, batch_size, hidden_size]
        out = out.reshape(out.shape[0], -1)
        # out_size = [batch_size, sequence_length * hidden_size]
        out = self.fc1(out)
        # out_size = [batch_size, 128]
        out = self.relu(out)
        out = self.fc2(out)
        # out_size = [batch_size, 64]
        out = self.relu(out)
        out = self.fc3(out)
        # out_size = [batch_size, 1]
        return out

class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, sequence_length, num_layers, dropout, device):
        super(BiRNN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size,
                          hidden_size, 
                          num_layers, 
                          batch_first = True, 
                          bidirectional = True, 
                          dropout = dropout)
        self.fc1 = nn.Linear(hidden_size * sequence_length * 2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x_size = [batch_size, sequence_length, features]
        h0 = torch.zeros(self.num_layers * 2, x.size()[0], self.hidden_size).to(self.device)
        out, hn = self.rnn(x, h0)
        # out_size = [batch_size, sequence_length, hidden_size * 2]
        # hn_size = [num_layers * 2, batch_size, hidden_size]
        out = out.reshape(out.shape[0], -1)
        # out_size = [batch_size, sequence_length * hidden_size * 2]
        out = self.fc1(out)
        # out_size = [batch_size, 256]
        out = self.relu(out)
        out = self.fc2(out)
        # out_size = [batch_size, 128]
        out = self.relu(out)
        out = self.fc3(out)
        # out_size = [batch_size, 64]
        out = self.relu(out)
        out = self.fc4(out)
        # out_size = [batch_size, 1]
        return out

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, sequence_length, num_layers, dropout, device):
        super(BiLSTM, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, 
                            hidden_size, 
                            num_layers, 
                            batch_first = True, 
                            bidirectional = True, 
                            dropout = dropout)
        self.fc1 = nn.Linear(hidden_size * sequence_length * 2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x_size = [batch_size, sequence_length, features]
        h0 = torch.zeros(self.num_layers*2, x.size()[0], self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers*2, x.size()[0], self.hidden_size).to(self.device)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        # out_size = [batch_size, sequence_length, hidden_size * 2]
        # hn_size = [num_layers * 2, batch_size, hidden_size]
        # cn_size = [num_layers * 2, batch_size, hidden_size]
        out = out.reshape(out.shape[0], -1)
        # out_size = [batch_size, sequence_length * hidden_size * 2]
        out = self.fc1(out)
        # out_size = [batch_size, 256]
        out = self.relu(out)
        out = self.fc2(out)
        # out_size = [batch_size, 128]
        out = self.relu(out)
        out = self.fc3(out)
        # out_size = [batch_size, 64]
        out = self.relu(out)
        out = self.fc4(out)
        # out_size = [batch_size, 1]
        return out

class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, sequence_length, num_layers, dropout, device):
        super(BiGRU, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, 
                          hidden_size, 
                          num_layers, 
                          batch_first = True, 
                          bidirectional = True, 
                          dropout = dropout)
        self.fc1 = nn.Linear(hidden_size * sequence_length * 2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x_size = [batch_size, sequence_length, features]
        h0 = torch.zeros(self.num_layers * 2, x.size()[0], self.hidden_size).to(self.device)
        out, hn = self.gru(x, h0)
        # out_size = [batch_size, sequence_length, hidden_size * 2]
        # hn_size = [num_layers * 2, batch_size, hidden_size]
        out = out.reshape(out.shape[0], -1)
        # out_size = [batch_size, sequence_length * hidden_size * 2]
        out = self.fc1(out)
        # out_size = [batch_size, 256]
        out = self.relu(out)
        out = self.fc2(out)
        # out_size = [batch_size, 128]
        out = self.relu(out)
        out = self.fc3(out)
        # out_size = [batch_size, 64]
        out = self.relu(out)
        out = self.fc4(out)
        # out_size = [batch_size, 1]
        return out

class RNN_encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, device):
        super(RNN_encoder, self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size = input_size, 
                          hidden_size = hidden_size, 
                          num_layers = num_layers, 
                          batch_first=True, 
                          dropout = dropout)
        
    def forward(self, x):
        # x_size = [batch_size, sequence_length, features]
        h0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size).to(self.device)
        out, hn = self.rnn(x, h0)
        # out_size = [batch_size, sequence_length, hidden_size]
        # hn_size = [num_layers, batch_size, hidden_size]
        return out, hn
        
class RNN_decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, device):
        super(RNN_decoder, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(input_size = input_size,
                          hidden_size = hidden_size,
                          num_layers = num_layers,
                          batch_first = True,
                          dropout = dropout)
        self.linear = nn.Linear(hidden_size, 1)
        
    def forward(self, x, hn):
        # x_size = [batch_size, features]
        x = x.unsqueeze(1)
        # x_size = [batch_size, 1, features]
        out, hn = self.rnn(x, hn)
        # out_size = [batch_size, 1, hidden_size]
        # hn_size = [num_layers, batch_size, hidden_size]
        out = self.linear(out)
        # out_size = [batch_size, 1, 1]
        return out, hn
        
class RNN_encoder_decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, device):
        super(RNN_encoder_decoder, self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.RNN_encoder = RNN_encoder(input_size = input_size, 
                                       hidden_size = hidden_size,
                                       num_layers = num_layers,
                                       dropout = dropout,
                                       device = device)
        
        self.RNN_decoder = RNN_decoder(input_size = input_size, 
                                       hidden_size = hidden_size,
                                       num_layers = num_layers,
                                       dropout = dropout,
                                       device = device)

    def forward(self, x, y, target_len, teacher_forcing_ratio, device):
        batch_size = x.shape[0]
        outputs = torch.zeros(batch_size, target_len, 1)
        _, hn = self.RNN_encoder(x)
        decoder_input = x[:,-1, :]
        #원하는 길이가 될 때까지 decoder를 실행한다.
        for t in range(target_len): 
            out, hn = self.RNN_decoder(decoder_input, hn)
            out =  out.squeeze(1)
            # out_size = [batch_size, 1]
            
            # teacher forcing을 구현한다.
            # teacher forcing에 해당하면 다음 인풋값으로는 예측한 값이 아니라 실제 값을 사용한다.
            if random.random() < teacher_forcing_ratio:
                decoder_output = y[:, t, :]
            else:
                decoder_output = out
            
            outputs[:,t,:] = out
            
            decoder_sub = torch.zeros(decoder_input.shape).to(device)
            decoder_sub[:, 0] = decoder_output[:, 0]
            decoder_sub[:, 1] = decoder_input[:, 0] - decoder_sub[:, 0]
            decoder_sub[:, 2] = decoder_input[:, 1] - decoder_sub[:, 1]
            decoder_input = decoder_sub
            
        # output_size = [batch_size, 7, 1]
        return outputs
        
class LSTM_encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, device):
        super(LSTM_encoder, self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size = input_size, 
                            hidden_size = hidden_size, 
                            num_layers = num_layers, 
                            batch_first=True, 
                            dropout = dropout)
        
    def forward(self, x):
        # x_size = [batch_size, sequence_length, features]
        h0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size).to(self.device)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        # out_size = [batch_size, sequence_length, hidden_size]
        # hn_size = [num_layers, batch_size, hidden_size]
        # cn_size = [num_layers, batch_size, hidden_size]
        return out, (hn, cn)

class LSTM_decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, device):
        super(LSTM_decoder, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size = input_size,
                            hidden_size = hidden_size,
                            num_layers = num_layers,
                            batch_first = True,
                            dropout = dropout)
        self.linear = nn.Linear(hidden_size, 1)
        
    def forward(self, x, hn, cn):
        # x_size = [batch_size, features]
        x = x.unsqueeze(1)
        # x_size = [batch_size, 1, features]
        out, (hn, cn) = self.lstm(x, (hn, cn))
        # out_size = [batch_size, 1, hidden_size]
        # hn_size = [num_layers, batch_size, hidden_size]
        # cn_size = [num_layers, batch_size, hidden_size]
        out = self.linear(out)
        # out_size = [batch_size, 1, 1]
        return out, (hn, cn)
        
class LSTM_encoder_decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, device):
        super(LSTM_encoder_decoder, self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.LSTM_encoder = LSTM_encoder(input_size = input_size, 
                                         hidden_size = hidden_size,
                                         num_layers = num_layers,
                                         dropout = dropout,
                                         device = device)
        
        self.LSTM_decoder = LSTM_decoder(input_size = input_size, 
                                         hidden_size = hidden_size,
                                         num_layers = num_layers,
                                         dropout = 0.3,
                                         device = device)

    def forward(self, x, y, target_len, teacher_forcing_ratio, device):
        batch_size = x.shape[0]
        outputs = torch.zeros(batch_size, target_len, 1)
        _, (hn, cn) = self.LSTM_encoder(x)
        decoder_input = x[:,-1, :]
        #원하는 길이가 될 때까지 decoder를 실행한다.
        for t in range(target_len): 
            out, (hn, cn) = self.LSTM_decoder(decoder_input, hn, cn)
            out =  out.squeeze(1)
            # out_size = [batch_size, 1]
            
            # teacher forcing을 구현한다.
            # teacher forcing에 해당하면 다음 인풋값으로는 예측한 값이 아니라 실제 값을 사용한다.
            if random.random() < teacher_forcing_ratio:
                decoder_output = y[:, t, :]
            else:
                decoder_output = out
                
            outputs[:,t,:] = out
            
            decoder_sub = torch.zeros(decoder_input.shape).to(device)
            decoder_sub[:, 0] = decoder_output[:, 0]
            decoder_sub[:, 1] = decoder_input[:, 0] - decoder_sub[:, 0]
            decoder_sub[:, 2] = decoder_input[:, 1] - decoder_sub[:, 1]
            decoder_input = decoder_sub

        # output_size = [batch_size, 7, 1]
        return outputs
        
class GRU_encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, device):
        super(GRU_encoder, self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size = input_size, 
                          hidden_size = hidden_size, 
                          num_layers = num_layers, 
                          batch_first=True, 
                          dropout = dropout)
        
    def forward(self, x):
        # x_size = [batch_size, sequence_length, features]
        h0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size).to(self.device)
        out, hn = self.gru(x, h0)
        # out_size = [batch_size, sequence_length, hidden_size]
        # hn_size = [num_layers, batch_size, hidden_size]
        return out, hn
        
class GRU_decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, device):
        super(GRU_decoder, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_size = input_size,
                          hidden_size = hidden_size,
                          num_layers = num_layers,
                          batch_first = True,
                          dropout = dropout)
        self.linear = nn.Linear(hidden_size, 1)
        
    def forward(self, x, hn):
        # x_size = [batch_size, features]
        x = x.unsqueeze(1)
        # x_size = [batch_size, 1, features]
        out, hn = self.gru(x, hn)
        # out_size = [batch_size, 1, hidden_size]
        # hn_size = [num_layers, batch_size, hidden_size]
        out = self.linear(out)
        # out_size = [batch_size, 1, 1]
        return out, hn
        
class GRU_encoder_decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, device):
        super(GRU_encoder_decoder, self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.GRU_encoder = GRU_encoder(input_size = input_size, 
                                       hidden_size = hidden_size,
                                       num_layers = num_layers,
                                       dropout = dropout,
                                       device = device)
        
        self.GRU_decoder = GRU_decoder(input_size = input_size, 
                                       hidden_size = hidden_size,
                                       num_layers = num_layers,
                                       dropout = 0.3,
                                       device = device)

    def forward(self, x, y, target_len, teacher_forcing_ratio, device):
        batch_size = x.shape[0]
        input_size = x.shape[2]
        outputs = torch.zeros(batch_size, target_len, 1)
        _, hn = self.GRU_encoder(x)
        decoder_input = x[:,-1, :]
        
        #원하는 길이가 될 때까지 decoder를 실행한다.
        for t in range(target_len): 
            out, hn = self.GRU_decoder(decoder_input, hn)
            out =  out.squeeze(1)
            # out_size = [batch_size, 1]
            
            # teacher forcing을 구현한다.
            # teacher forcing에 해당하면 다음 인풋값으로는 예측한 값이 아니라 실제 값을 사용한다.
            if random.random() < teacher_forcing_ratio:
                decoder_output = y[:, t, :]
            else:
                decoder_output = out
                
            outputs[:,t,:] = out
            
            decoder_sub = torch.zeros(decoder_input.shape).to(device)
            decoder_sub[:, 0] = decoder_output[:, 0]
            decoder_sub[:, 1] = decoder_input[:, 0] - decoder_sub[:, 0]
            decoder_sub[:, 2] = decoder_input[:, 1] - decoder_sub[:, 1]
            decoder_input = decoder_sub

        # output_size = [batch_size, 7, 1]
        return outputs
        
class BiRNN_encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, device):
        super(BiRNN_encoder, self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size = input_size, 
                          hidden_size = hidden_size, 
                          num_layers = num_layers, 
                          batch_first=True, 
                          bidirectional = True, 
                          dropout = dropout)
        
    def forward(self, x):
        # x_size = [batch_size, sequence_length, features]
        h0 = torch.zeros(self.num_layers * 2, x.size()[0], self.hidden_size).to(self.device)
        out, hn = self.rnn(x, h0)
        # out_size = [batch_size, sequence_length, hidden_size * 2]
        # hn_size = [num_layers * 2, batch_size, hidden_size]
        return out, hn
        
class BiRNN_decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, device):
        super(BiRNN_decoder, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(input_size = input_size,
                          hidden_size = hidden_size,
                          num_layers = num_layers*2,
                          batch_first = True,
                          dropout = dropout)
        self.linear = nn.Linear(hidden_size, 1)
        
    def forward(self, x, hn):
        # x_size = [batch_size, features]
        x = x.unsqueeze(1)
        # x_size = [batch_size, 1, features]
        out, hn = self.rnn(x, hn)
        # out_size = [batch_size, 1, hidden_size]
        # hn_size = [num_layers * 2, batch_size, hidden_size]
        out = self.linear(out)
        # out_size = [batch_size, 1, 1]
        return out, hn
        
class BiRNN_encoder_decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, device):
        super(BiRNN_encoder_decoder, self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.BiRNN_encoder = BiRNN_encoder(input_size = input_size, 
                                           hidden_size = hidden_size,
                                           num_layers = num_layers,
                                           dropout = dropout,
                                           device = device)
        
        self.BiRNN_decoder = BiRNN_decoder(input_size = input_size, 
                                           hidden_size = hidden_size,
                                           num_layers = num_layers,
                                           dropout = dropout,
                                           device = device)

    def forward(self, x, y, target_len, teacher_forcing_ratio, device):
        batch_size = x.shape[0]
        outputs = torch.zeros(batch_size, target_len, 1)
        _, hn = self.BiRNN_encoder(x)
        decoder_input = x[:,-1, :]
        
        #원하는 길이가 될 때까지 decoder를 실행한다.
        for t in range(target_len): 
            out, hn = self.BiRNN_decoder(decoder_input, hn)
            out =  out.squeeze(1)
            # out_size = [batch_size, 1]
            
            # teacher forcing을 구현한다.
            # teacher forcing에 해당하면 다음 인풋값으로는 예측한 값이 아니라 실제 값을 사용한다.
            if random.random() < teacher_forcing_ratio:
                decoder_output = y[:, t, :]
            else:
                decoder_output = out
                
            outputs[:,t,:] = out
            
            decoder_sub = torch.zeros(decoder_input.shape).to(device)
            decoder_sub[:, 0] = decoder_output[:, 0]
            decoder_sub[:, 1] = decoder_input[:, 0] - decoder_sub[:, 0]
            decoder_sub[:, 2] = decoder_input[:, 1] - decoder_sub[:, 1]
            decoder_input = decoder_sub

        # output_size = [batch_size, 7, 1]
        return outputs
        
class BiLSTM_encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, device):
        super(BiLSTM_encoder, self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size = input_size, 
                            hidden_size = hidden_size, 
                            num_layers = num_layers, 
                            batch_first=True, 
                            bidirectional = True, 
                            dropout = dropout)
        
    def forward(self, x):
        # x_size = [batch_size, sequence_length, features]
        h0 = torch.zeros(self.num_layers * 2, x.size()[0], self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers * 2, x.size()[0], self.hidden_size).to(self.device)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        # out_size = [batch_size, sequence_length, hidden_size * 2]
        # hn_size = [num_layers * 2, batch_size, hidden_size]
        # cn_size = [num_layers * 2, batch_size, hidden_size]
        return out, (hn, cn)

class BiLSTM_decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, device):
        super(BiLSTM_decoder, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size = input_size,
                            hidden_size = hidden_size,
                            num_layers = num_layers*2,
                            batch_first = True,
                            dropout = dropout)
        self.linear = nn.Linear(hidden_size, 1)
        
    def forward(self, x, hn, cn):
        # x_size = [batch_size, features]
        x = x.unsqueeze(1)
        # x_size = [batch_size, 1, features]
        out, (hn, cn) = self.lstm(x, (hn, cn))
        # out_size = [batch_size, 1, hidden_size]
        # hn_size = [num_layers * 2, batch_size, hidden_size]
        # cn_size = [num_layers * 2, batch_size, hidden_size]
        out = self.linear(out)
        # out_size = [batch_size, 1, 1]
        return out, (hn, cn)
        
class BiLSTM_encoder_decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, device):
        super(BiLSTM_encoder_decoder, self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.BiLSTM_encoder = BiLSTM_encoder(input_size = input_size, 
                                             hidden_size = hidden_size,
                                             num_layers = num_layers,
                                             dropout = dropout,
                                             device = device)
        
        self.BiLSTM_decoder = BiLSTM_decoder(input_size = input_size, 
                                             hidden_size = hidden_size,
                                             num_layers = num_layers,
                                             dropout = dropout,
                                             device = device)

    def forward(self, x, y, target_len, teacher_forcing_ratio, device):
        batch_size = x.shape[0]
        input_size = x.shape[2]
        outputs = torch.zeros(batch_size, target_len, 1)
        _, (hn, cn) = self.BiLSTM_encoder(x)
        decoder_input = x[:,-1, :]
        
        #원하는 길이가 될 때까지 decoder를 실행한다.
        for t in range(target_len): 
            out, (hn, cn) = self.BiLSTM_decoder(decoder_input, hn, cn)
            out =  out.squeeze(1)
            # out_size = [batch_size, 1]
            
            # teacher forcing을 구현한다.
            # teacher forcing에 해당하면 다음 인풋값으로는 예측한 값이 아니라 실제 값을 사용한다.
            if random.random() < teacher_forcing_ratio:
                decoder_output = y[:, t, :]
            else:
                decoder_output = out
                
            outputs[:,t,:] = out
            
            decoder_sub = torch.zeros(decoder_input.shape).to(device)
            decoder_sub[:, 0] = decoder_output[:, 0]
            decoder_sub[:, 1] = decoder_input[:, 0] - decoder_sub[:, 0]
            decoder_sub[:, 2] = decoder_input[:, 1] - decoder_sub[:, 1]
            decoder_input = decoder_sub

        # output_size = [batch_size, 7, 1]
        return outputs
        
class BiGRU_encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, device):
        super(BiGRU_encoder, self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size = input_size, 
                          hidden_size = hidden_size, 
                          num_layers = num_layers, 
                          batch_first=True, 
                          bidirectional = True, 
                          dropout = dropout)
        
    def forward(self, x):
        # x_size = [batch_size, sequence_length, features]
        h0 = torch.zeros(self.num_layers * 2, x.size()[0], self.hidden_size).to(self.device)
        out, hn = self.gru(x, h0)
        # out_size = [batch_size, sequence_length, hidden_size * 2]
        # hn_size = [num_layers * 2, batch_size, hidden_size]
        return out, hn
        
class BiGRU_decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, device):
        super(BiGRU_decoder, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_size = input_size,
                          hidden_size = hidden_size,
                          num_layers = num_layers*2,
                          batch_first = True,
                          dropout = dropout)
        self.linear = nn.Linear(hidden_size, 1)
        
    def forward(self, x, hn):
        # x_size = [batch_size, features]
        x = x.unsqueeze(1)
        # x_size = [batch_size, 1, features]
        out, hn = self.gru(x, hn)
        # out_size = [batch_size, 1, hidden_size]
        # hn_size = [num_layers * 2, batch_size, hidden_size]
        out = self.linear(out)
        # out_size = [batch_size, 1, 1]
        return out, hn
        
class BiGRU_encoder_decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, device):
        super(BiGRU_encoder_decoder, self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.BiGRU_encoder = BiGRU_encoder(input_size = input_size, 
                                           hidden_size = hidden_size,
                                           num_layers = num_layers,
                                           dropout = dropout,
                                           device = device)
        
        self.BiGRU_decoder = BiGRU_decoder(input_size = input_size, 
                                           hidden_size = hidden_size,
                                           num_layers = num_layers,
                                           dropout = dropout,
                                           device = device)

    def forward(self, x, y, target_len, teacher_forcing_ratio, device):
        batch_size = x.shape[0]
        input_size = x.shape[2]
        outputs = torch.zeros(batch_size, target_len, 1)
        _, hn = self.BiGRU_encoder(x)
        decoder_input = x[:,-1, :]
        
        #원하는 길이가 될 때까지 decoder를 실행한다.
        for t in range(target_len): 
            out, hn = self.BiGRU_decoder(decoder_input, hn)
            out =  out.squeeze(1)
            # out_size = [batch_size, 1]
            
            # teacher forcing을 구현한다.
            # teacher forcing에 해당하면 다음 인풋값으로는 예측한 값이 아니라 실제 값을 사용한다.
            if random.random() < teacher_forcing_ratio:
                decoder_output = y[:, t, :]
            else:
                decoder_output = out
                
            outputs[:,t,:] = out
            
            decoder_sub = torch.zeros(decoder_input.shape).to(device)
            decoder_sub[:, 0] = decoder_output[:, 0]
            decoder_sub[:, 1] = decoder_input[:, 0] - decoder_sub[:, 0]
            decoder_sub[:, 2] = decoder_input[:, 1] - decoder_sub[:, 1]
            decoder_input = decoder_sub

        # output_size = [batch_size, 7, 1]
        return outputs