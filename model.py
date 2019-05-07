import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


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

# self-defined loss function to handle 
# vector labels
class x_entropy_loss(nn.Module):
    def __init__(self):
        super(x_entropy_loss, self).__init__()

    def forward(self, pred, target):
        pred = F.log_softmax(pred, dim=1)
        loss = pred * target
        return -torch.sum(loss) / target.shape[0]


# Main Model
class emotionDetector(nn.Module):
    def __init__(self, lstm_dim, n_classes, lstm_drop_rate, text_drop_rate, batch_size):
        super(emotionDetector, self).__init__()

        self.text_cnn = textCNN(text_drop_rate)
        self.hidden_dim = lstm_dim
        self.gru = nn.GRU(768, self.hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
    
        self.dr = nn.Linear(768*2, 512)

        self.batch_size = batch_size
        self.fc = nn.Linear(2* lstm_dim, n_classes)
        nn.init.xavier_uniform_(self.fc.weight)
        self.hidden = self.init_hidden()
        self.softmax = nn.Softmax(dim=1)
        self.drop_rate = lstm_drop_rate

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
        return (autograd.Variable(torch.randn(2, self.batch_size, self.hidden_dim).cuda()))
                #autograd.Variable(torch.randn(2, 1, self.hidden_dim).cuda()))

    def forward(self, x, seq_len):
        n_batch, n_utt, n_channel, n_word, dim_size = x.shape

        ################ CNN ######################
        #seq_x = x[:, :,  1:, :]  # sequence embedding
        sent_x = x[:, :, :, 0, :]  # sentence embedding
        sent_x = sent_x.view(n_batch, n_utt, -1)
        #seq_x = self.text_cnn(seq_x)

        #x = torch.cat((seq_x, sent_x), 1)
        #x = F.dropout(x, 0.1, training=self.training)
        #x = self.dr(x)
        x = sent_x

        ############### RNN ####################
        # padding
        seq_len = torch.Tensor(seq_len)
        seq_len = seq_len.cuda()
        _, idx_sort = torch.sort(seq_len, dim=0, descending=True)
        idx_sort = idx_sort.cuda()
        _, idx_unsort = torch.sort(idx_sort, dim=0, descending=False)
        idx_unsort = idx_unsort.cuda()
        sequence = x.index_select(0, idx_sort)
        lengths = list(seq_len[idx_sort])
        seq_packed = nn.utils.rnn.pack_padded_sequence(input=sequence, lengths=lengths, batch_first=True)

        h0= self.init_hidden()

        x = x.view(n_batch, n_utt, -1)
        gru_out, hn = self.gru(seq_packed, h0)

        x_padded = nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True)
        x = x_padded[0].index_select(0, idx_unsort) # padded sequence

        x = self.unrolling(x, seq_len.tolist())

        x = self.fc(x)

        x = F.dropout(x, self.drop_rate, training=self.training)

        scores = x
            
        return scores

