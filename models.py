# Model classes

import torch.nn as nn
import torch


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


class Net(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass
