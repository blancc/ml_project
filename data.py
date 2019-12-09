# Dataset classes
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import torch
import torch.functional as F
from setup import MODEL, SUBSAMPLE, WORD_LENGTH


class SignalDataset(Dataset):
    def __init__(self, sub_sample=1):
        self.sub_sample = sub_sample
        self.rolloffs = [1, 2]  # [1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.n_batch = 500
        if MODEL == 'Conv2D':
            import torchaudio
            self.fft = torchaudio.transforms.Spectrogram(n_fft=127, win_length=4)

    def __getitem__(self, index):
        ro_index = index // self.n_batch
        batch = index % self.n_batch + 1

        with open(f'dataset/SNR10/QPSK/wvform_rx_real_rollOff{self.rolloffs[ro_index]}_batch{batch}', 'r') as f:
            X = [line.strip(' \n') for line in f]
        X = np.array(X[1::self.sub_sample], dtype=np.float64)
        X = torch.from_numpy(X).float()

        with open(f'dataset/SNR10/QPSK/sym_tx_rollOff{self.rolloffs[ro_index]}_batch{batch}', 'r') as f:
            y = [line.strip(' \n').split() for line in f]
        y = np.array(y[1::], dtype=np.float64)
        y = torch.from_numpy(y).float()

        if MODEL == 'Conv2D':
            return self.fft(X.unsqueeze(0)), y
        if MODEL == 'Conv1D':
            return X.unsqueeze(0), y
        if MODEL == 'MLP':
            return X, y*4
        if MODEL == 'Trans':
            return X.view(WORD_LENGTH, -1), y

    def __len__(self):
        return len(self.rolloffs)*self.n_batch


def get_loaders(batch_size, split_percent=0.8):
    dataset = SignalDataset(sub_sample=SUBSAMPLE)

    train_size = int(split_percent * len(dataset))
    test_size = len(dataset) - train_size

    train_set, test_set = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    return train_loader, test_loader
