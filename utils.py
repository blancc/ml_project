# Misc.: dataset preprocessing, data visualization, ...
import numpy as np
import matplotlib.pyplot as plt


def load_data(snr, code, index):
    pass


if __name__ == "__main__":
    with open('dataset/SNR5/QPSK/wvform_rx_real_rollOff1', 'r') as f:
        real = [line.strip(' \n') for line in f]
        real = np.array(real[1::], dtype=np.float64)

    with open('dataset/SNR5/QPSK/wvform_rx_imag_rollOff1', 'r') as f:
        imag = [line.strip(' \n') for line in f]
        imag = np.array(imag[1::], dtype=np.float64)

    plt.scatter(real, imag, marker='x')
    plt.show()
