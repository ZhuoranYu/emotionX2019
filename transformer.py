import torch
import torch.nn as nn 
import copy
import math 
import numpy as np 
import torch.nn.functional as F
from torch.autograd import Variable

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=24):
        super().__init__()
        self.d_model = d_model
        
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
                
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        x = x + Variable(self.pe[:,:seq_len], \
        requires_grad=False).cuda()
        return x




class MultiHeadAttention(nn.Module):
    def __init__(self, heads, dim, dropout=0.1):
        super().__init__()

        self.dim = dim 
        self.d_k = dim // heads
        self.heads = heads 

        self.q_linear = nn.Linear(dim, dim)
        self.v_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout) 
        self.out = nn.Linear(dim, dim)

    def forward(self, q, k, v, mask=None):
        # TODO check for input shape
        bs = 1

        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.heads, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.heads, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.heads, self.d_k)

        # transpose to get dimensions (bs, heads, seq_len, dim)
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using attention funciton
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concage heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous()\
                 .view(bs, -1, self.dim)

        output = self.out(concat)
        return output

def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1))
    scores *= (1 / math.sqrt(d_k))
    if mask is not None:
        #mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask==0, -1e9)
    scores = F.softmax(scores, dim=1)
    if dropout is not None:
        scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output

class Norm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()

        self.dim = dim
        self.alpha = nn.Parameter(torch.ones(self.dim))
        self.bias = nn.Parameter(torch.zeros(self.dim))
        self.eps = eps 
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=1, keepdim=True)) \
                / (x.std(dim=1, keepdim=True)) + self.eps + self.bias
        return norm 


# May Vary Number of Encoder Decoders
# currently 2
class EncoderLayer(nn.Module):
    def __init__(self, dim, heads, dropout=0.1):
        super().__init__()
        self.norm1 = Norm(dim)
        self.norm2 = Norm(dim)
        self.attn = MultiHeadAttention(heads, dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm1(x)
        x2 = self.attn(x2, x2, x2, mask)
        x = x + self.dropout1(x2)

        x2 = self.norm2(x)
        x2 = self.attn(x2, x2, x2, mask)
        x = x + self.dropout2(x2)

        return x 

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Encoder(nn.Module):
    def __init__(self, dim, N, heads):
        super().__init__()
        self.N = N # number of encoder and decoders
        self.pe = PositionalEncoder(dim)
        self.layers = get_clones(EncoderLayer(dim, heads), N)
        self.norm = Norm(dim)

    def forward(self, x, mask):
        # x should be output from residual GRU
        x = self.pe(x)
        
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        # one-hidden-dim network
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        scores = self.fc2(x)

        return scores

class Transformer(nn.Module):
    # TODO unify dimension size for transformer
    def __init__(self, dim=1536, N=6, heads=12, max_seq_len=24):
        super().__init__()
        self.dim = dim
        self.scale_factor = math.sqrt(dim)
        self.max_seq_len = max_seq_len

        self.encoder = Encoder(dim, N, heads)


    def forward(self, x):
        _, n_utt, _ = x.shape
        len_mask = np.ones((self.max_seq_len, 1))
        len_mask[n_utt:] = 0
        attn_mask = np.matmul(len_mask, len_mask.transpose())
        attn_mask = torch.from_numpy(attn_mask)
        attn_mask = torch.eq(attn_mask, 0).cuda() if torch.cuda.is_available() else torch.eq(attn_mask, 0)

        # TODO check SRC_MASK: what does unsqueeze do

        e_outputs = self.encoder(x, attn_mask)
        output = e_outputs.view(self.max_seq_len, -1)

        return output
