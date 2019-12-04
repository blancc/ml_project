# Misc.: dataset preprocessing, data visualization, ...
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchaudio
from tqdm import tqdm


def load_data(snr, code, index):
    pass


def compute_accuracy(model, loader):
    score = 0
    length = 0
    model.eval()

    for X, y in tqdm(loader):
        X = X.to('cuda')
        y = y.to('cuda')

        Y = model(X)

        # Y[Y >= 0] = 1. # Test with or without
        # Y[Y < 0] = -1.
        y = torch.flatten(y, 1)

        score += (Y == y).sum()
        print(score)
        length += len(y)

    return score / length


if __name__ == "__main__":
    with open('dataset/SNR10/QPSK/wvform_rx_real_rollOff1_batch1', 'r') as f:
        real = [line.strip(' \n') for line in f]
        real = np.array(real[1::], dtype=np.float64)

    with open(f'dataset/SNR10/QPSK/sym_tx_rollOff1_batch1', 'r') as f:
        y = [line.strip(' \n').split() for line in f]
        y = np.array(y[1::], dtype=np.float64)

    specgram = torchaudio.transforms.Spectrogram(n_fft=15)(torch.from_numpy(real).unsqueeze(0))

    print("Shape of spectrogram: {}".format(specgram.size()))

    plt.figure()
    plt.imshow(specgram.log2()[0, :, :].numpy())
    plt.show()
