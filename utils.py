# Misc.: dataset preprocessing, data visualization, ...
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchaudio
from setup import DEVICE, SUBSAMPLE, TARGET_LENGTH, SIGNAL_LENGTH
from tqdm import tqdm
import torch.functional as F


def visualize_predictions(model, dataloader, denoiser=None):
    model.eval()
    with torch.no_grad():
        t = torch.Tensor()
        batch = next(iter(dataloader))
        X, y = batch
        model.cpu()

        if denoiser:
            Y = denoiser(X)
        Y = model(Y)

        model.to(DEVICE)
        Y = Y.detach().numpy()
        plt.clf()
        plt.scatter(Y[:, :TARGET_LENGTH//2], Y[:, TARGET_LENGTH//2:], marker="x")
        plt.scatter(y[:, 0], y[:, 1], marker="o")
        return plt


def visualize_noise(model, dataloader):
    model.eval()
    with torch.no_grad():
        t = torch.Tensor()
        batch = next(iter(dataloader))
        X, y = batch
        model.cpu()

        Y = model(X)

        model.to(DEVICE)
        Y = Y.detach().numpy()
        plt.clf()
        plt.plot(np.arange(SIGNAL_LENGTH//SUBSAMPLE), Y[0])
        plt.plot(np.arange(SIGNAL_LENGTH//SUBSAMPLE), y[0])
        return plt


def compute_accuracy(model, loader, denoiser=None):
    score = 0
    length = 0
    model.eval()
    with torch.no_grad():
        for X, y in tqdm(loader):
            X = X.to(DEVICE)
            y = y.to(DEVICE)

            if denoiser:
                Y = denoiser(X)
            Y = model(Y)
            Y[Y >= 0] = 1.
            Y[Y < 0] = -1.
            y = torch.flatten(y, 1)
            score += (Y.cpu() == y.cpu()).sum().item()
            length += y.numel()
        return score / length


if __name__ == "__main__":
    with open('dataset/SNR10/QPSK/wvform_rx_real_rollOff1_batch1', 'r') as f:
        real = [line.strip(' \n') for line in f]
        real = np.array(real[1::SUBSAMPLE], dtype=np.float64)
        real = real/abs(real).max()

    with open(f'dataset/SNR10/QPSK/sym_tx_rollOff1_batch1', 'r') as f:
        y = [line.strip(' \n').split() for line in f]
        y = np.array(y[1::], dtype=np.float64)

    specgram = torchaudio.transforms.Spectrogram(
        n_fft=127, win_length=4)(torch.from_numpy(real).unsqueeze(0))

    print("Shape of spectrogram: {}".format(specgram.size()))

    plt.figure()
    # plt.imshow(specgram.log2()[0, :, :].numpy())
    plt.plot(real)
    plt.show()
