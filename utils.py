# Misc.: dataset preprocessing, data visualization, ...
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchaudio
from setup import DEVICE
from tqdm import tqdm


def visualize_predictions(model, dataloader):
    model.eval()
    t = torch.Tensor()
    for batch in dataloader:
        X, y = batch
        model.cpu()

        Y = model(X)
        t = torch.cat((t, Y.cpu()))
    print("AAA")
    t = t.view(2, -1)
    t = t.detach().numpy()
    fig, ax = plt.subplots()
    ax.scatter(t[0], t[1], marker="x")
    return fig


def compute_accuracy(model, loader):
    score = 0
    length = 0
    model.eval()
    for X, y in tqdm(loader):
        X = X.to(DEVICE)
        y = y.to(DEVICE)

        Y = model(X)
        Y[Y >= 0] = 1.
        Y[Y < 0] = -1.
        y = torch.flatten(y, 1)
        score += (Y.cpu() == y.cpu()).sum().item()
        length += y.numel()
    return score / length


if __name__ == "__main__":
    with open('dataset/SNR10/QPSK/wvform_rx_real_rollOff1_batch1', 'r') as f:
        real = [line.strip(' \n') for line in f]
        real = np.array(real[1::], dtype=np.float64)

    with open(f'dataset/SNR10/QPSK/sym_tx_rollOff1_batch1', 'r') as f:
        y = [line.strip(' \n').split() for line in f]
        y = np.array(y[1::], dtype=np.float64)

    specgram = torchaudio.transforms.Spectrogram(
        n_fft=127, win_length=4)(torch.from_numpy(real).unsqueeze(0))

    print("Shape of spectrogram: {}".format(specgram.size()))

    plt.figure()
    plt.imshow(specgram.log2()[0, :, :].numpy())
    plt.show()
