# Misc.: dataset preprocessing, data visualization, ...
import torch
import numpy as np
import matplotlib.pyplot as plt


def load_data(snr, code, index):
    pass


def compute_accuracy(model, loader):
    score = 0
    length = 0
    model.eval()

    for X, y in loader:
        y.view(-1)

        Y = model(x)

        Y[Y >= 0] = 1.
        Y[Y < 0] = -1.

        score += (Y == y).sum()
        length += len(y)

    return score / length


if __name__ == "__main__":
    # with open('dataset/SNR7/QPSK/wvform_rx_real_rollOff9', 'r') as f:
    #     real = [line.strip(' \n') for line in f]
    #     real = np.array(real[1::], dtype=np.float64)

    # with open('dataset/SNR7/QPSK/wvform_rx_imag_rollOff9', 'r') as f:
    #     imag = [line.strip(' \n') for line in f]
    #     imag = np.array(imag[1::], dtype=np.float64)

    with open('dataset/SNR7/QPSK/wvform_rx_real_rollOff9', 'r') as f:
        real = [line.strip(' \n') for line in f]
        real = np.array(real[1::], dtype=np.float64)

    plt.plot(real)
    plt.show()
