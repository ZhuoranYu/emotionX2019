import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


class textCNN(nn.Module):
    def __init__(self):
        super(textCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3)
        self.pool2 = nn.MaxPool1d(kernel_size=3)

        self.conv3 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=4)
        self.pool3 = nn.MaxPool1d(kernel_size=4)



# self-defined loss function to handle 
# vector labels
class x_entropy_loss(nn.Module):
    def __init__(self):
        super(x_entropy_loss, self).__init__()

    def forward(self, pred, target):
        logged_pred = torch.log(pred)
        loss = -torch.sum(target * logged_pred)
        return loss


# Main Model
class emotionDetector(nn.Module):
    def __init__(self, lstm_dim, n_classes, lstm_drop_rate, text_drop_rate):
        super(emotionDetector, self).__init__()

        self.text_cnn = charCNN(text_drop_rate)
        self.hidden_dim = lstm_dim
        self.lstm = nn.LSTM(512, self.hidden_dim, num_layers=3, bidirectional=True)
     
        self.fc = nn.Linear(2* lstm_dim, n_classes)
        self.hidden = self.init_hidden()
        self.softmax = nn.Softmax(dim=1)
        self.drop_rate = lstm_drop_rate

    def init_hidden(self):
        return (autograd.Variable(torch.randn(6, 1, self.hidden_dim).cuda()),
                autograd.Variable(torch.randn(6, 1, self.hidden_dim).cuda()))

    def forward(self, x):
        N = x.shape[0]
        x = self.char_cnn(x)
        x = F.relu(x)
        h0, c0 = self.init_hidden()

        x = x.view(N, 1, -1)
        lstm_out, (hn, cn) = self.lstm(x.view(N, 1, -1), (h0, c0))
        lstm_out = F.dropout(lstm_out, self.drop_rate, training=self.training)
        x = self.fc(lstm_out.view(N, -1))

        x = F.dropout(x, self.drop_rate, training=self.training)

        scores = self.softmax(x)
            
        return scores

