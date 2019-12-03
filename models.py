# Model classes

import torch.nn as nn
import torch
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.pow(1.e4, torch.arange(
            0, d_model, 2).float() / d_model)
        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


class Transformer(nn.Module):
    def __init__(self, d_model, n_head, max_len, dim_feedforward=2048, dropout=0.1,
                 n_layers=6, device='cpu'):
        super().__init__()
        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len+1)
        self.max_len = max_len
        layers = nn.TransformerEncoderLayer(d_model, n_head,
                                            dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(layers, n_layers)
        self.device = device

    def forward(self, x, lengths):
        mask = []
        for i in range(len(lengths)):
            lengths = lengths.squeeze(0)
            mask_v = torch.zeros(self.max_len+1)
            mask_v[:lengths[i]] = 1
            mask.append(mask_v)
        mask = torch.cat(mask).to(self.device)

        out = torch.zeros(self.max_len + 1, x.size(1), x.size(2)).to(self.device)
        out[:x.size(0), :, :] = x
        out = self.pos_encoder(out)
        out = self.transformer_encoder(out, mask=mask)

        return out


class Conv2D(nn.Module):
    def __init__(self, in_dim, out_dim):
        self.conv1 = nn.Conv2d(in_dim, out_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(in_dim, out_dim, 3, padding=1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        return out


class Conv1D(nn.Module):
    def __init__(self, in_dim, out_dim):
        self.conv1 = nn.Conv1d(in_dim, out_dim, 3, padding=1)
        self.conv2 = nn.Conv1d(in_dim, out_dim, 3, padding=1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.max_pool1d(out, 2)
        return out


class Net(nn.Module):
    def __init__(self, model, dropout=0.4):
        self.name = model
        self.dropout = nn.Dropout(dropout)

        if model == "Conv1D":
            self.block1 = Conv1D(TODO, TODO)
            self.block2 = Conv1D(TODO, TODO)

        if model == "Conv2D":
            self.block1 = Conv2D(TODO, TODO)
            self.block2 = Conv2D(TODO, TODO)

        self.fc1 = nn.Linear(TODO, TODO)
        self.fc2 = nn.Linear(TODO, TODO)
        self.fc3 = nn.Linear(TODO, TODO)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.dropout(F.relu(self.fc1(out)))
        out = self.dropout(F.relu(self.fc2(out)))
        out = self.dropout(F.relu(self.fc3(out)))

        out[out >= 0] = 1
        out[out < 0] = -1

        return out
