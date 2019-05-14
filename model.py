import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.parameter import Parameter
import math

from transformer import Transformer

class textCNN(nn.Module):
    def __init__(self, drop_rate):
        super(textCNN, self).__init__()

        self.drop_rate = drop_rate

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(2, 768), padding=0)
        self.pool1 = nn.MaxPool1d(kernel_size=14)

        self.conv2 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(3, 768))
        self.pool2 = nn.MaxPool1d(kernel_size=13)

        self.conv3 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(4, 768))
        self.pool3 = nn.MaxPool1d(kernel_size=12)

        #nn.init.xavier_uniform(self.conv1.weight)
        #nn.init.xavier_uniform(self.conv2.weight)
        #nn.init.xavier_uniform(self.conv3.weight)

    def forward(self, x):
        N = x.shape[0]
        x1 = self.conv1(x)
        x1 = x1.view(N, 256, -1)
        x1 = F.relu(x1)
        x1 = self.pool1(x1)

        x2 = self.conv2(x)
        x2 = x2.view(N, 256, -1)
        x2 = F.relu(x2)
        x2 = self.pool2(x2)

        x3 = self.conv3(x)
        x3 = x3.view(N, 256, -1)
        x3 = F.relu(x3)
        x3 = self.pool3(x3)

        x = [x1, x2, x3]
        result = torch.cat(x, 1)
        result = F.dropout(result, self.drop_rate, training=self.training)

        result = result.view(N, -1)
        return result

class resLSTM(nn.Module):
    def __init__(self, lstm_dim, lstm_drop_rate):
        super(resLSTM, self).__init__()
        self.hidden_dim = lstm_dim
        self.gru1 = nn.GRU(768*2, self.hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.gru2 = nn.GRU(768*2, self.hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.gru3 = nn.GRU(768*2, self.hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.gru4 = nn.GRU(768*2, self.hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.gru5 = nn.GRU(768*2, self.hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.gru6 = nn.GRU(768*2, self.hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.drop_rate = lstm_drop_rate

        self.fc1 = nn.Linear(self.hidden_dim * 2, 768*2)
        self.fc2 = nn.Linear(self.hidden_dim * 2, 768*2)
        self.fc3 = nn.Linear(self.hidden_dim * 2, 768*2)
        self.fc4 = nn.Linear(self.hidden_dim * 2, 768*2)
        self.fc5 = nn.Linear(self.hidden_dim * 2, 768*2)
        self.fc6 = nn.Linear(self.hidden_dim * 2, 768*2)

    def init_hidden(self):
        return (autograd.Variable(torch.randn(2, 1, self.hidden_dim).cuda()))

    def forward(self, x):
        h10 = self.init_hidden()
        h20 = self.init_hidden()
        h30 = self.init_hidden()
        h40 = self.init_hidden()
        gru_out1, h1n = self.gru1(x)
        gru_out1 = self.fc1(gru_out1)
        x = F.relu(gru_out1) + x # res connection

        gru_out2, h2n = self.gru2(x)
        gru_out2 = self.fc2(gru_out2)
        x = F.relu(gru_out2) + x # res connection

        gru_out3, h3n = self.gru3(x)
        gru_out3 = self.fc3(gru_out3)
        x = F.relu(gru_out3) + x # res connection

        gru_out4, h4n = self.gru4(x)
        gru_out4 = self.fc4(gru_out4)
        x = F.relu(gru_out4) + x

        gru_out5, h5n = self.gru5(x)
        gru_out5 = self.fc5(gru_out5)
        x = F.relu(gru_out5) + x

        gru_out6, h6n = self.gru6(x)
        gru_out6 = self.fc6(gru_out6)
        x = F.relu(gru_out6) + x

        x = F.dropout(x, self.drop_rate, training=self.training)
        return x

# Main Model
class emotionDetector(nn.Module):
    def __init__(self, lstm_dim, n_classes, lstm_drop_rate, text_drop_rate, batch_size):
        super(emotionDetector, self).__init__()

        self.text_cnn = textCNN(text_drop_rate)
        self.hidden_dim = lstm_dim
        self.max_seq_len = 24
        self.resLSTM = resLSTM(self.hidden_dim, lstm_drop_rate)
        self.batch_size = batch_size
        self.fc = nn.Linear(768*2, n_classes)
        nn.init.xavier_uniform_(self.fc.weight)

        self.trans = Transformer(dim=768*2)

    def unrolling(self, x, lengths):
        result = None
        for idx, seq_len in enumerate(lengths):
            for i in range(0, int(seq_len)):
                next_word = x[idx, i, :]
                next_word = next_word.view(1, -1)
                if result is None:
                    result = next_word
                else:
                    result = torch.cat([result, next_word], 0)
        return result


    def init_hidden(self):
        return (autograd.Variable(torch.randn(2, 1, self.hidden_dim).cuda()))
                #autograd.Variable(torch.randn(2, 1, self.hidden_dim).cuda()))

    def forward(self, x):
        n_utt, n_channel, n_word, dim_size = x.shape

        ################ CNN ######################
        seq_x = x[:, :,  1:, :]  # sequence embedding
        sent_x = x[:, :, 0, :]  # sentence embedding
        sent_x = sent_x.view(n_utt, -1)
        seq_x = self.text_cnn(seq_x)

        x = torch.cat((seq_x, sent_x), 1)
        x = x.view(1, n_utt, -1)

        ############### RNN ####################
        x = self.resLSTM(x)

        ############### TRANSFORMER ENCODER #########
        res = self.max_seq_len - n_utt
        if res != 0:
            padding = torch.zeros(1, res, 1536).cuda()
            x = torch.cat([x, padding], 1)
        x = self.trans(x)

        x = self.fc(x)

        x = x.view(self.max_seq_len, -1)
        x = x[:n_utt, :]

        scores = x
            
        return scores

