# Dataset classes
from torch.utils.data import Dataset, DataLoader, random_split


class SignalDataset(Dataset):
    def __init__(self, sub_sample=1):
        self.sub_sample = sub_sample
        self.rolloffs = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.n_batch = 500

    def __getitem__(self, index):
        ro_index = index % self.n_batch
        batch = index // self.n_batch

        with open(f'dataset/SNR10/QPSK/wvform_rx_real_rollOff{self.rolloffs[ro_index]}_batch{batch}', 'r') as f:
            X = [line.strip(' \n') for line in f]
            X = np.array(X[1::self.sub_sample], dtype=np.float64)

        with open(f'dataset/SNR10/QPSK/sym_tx_rollOff{self.rolloffs[ro_index]}_batch{batch}', 'r') as f:
            y = [line.strip(' \n').split() for line in f]
            y = np.array(y[1::self.sub_sample], dtype=np.float64)

        return torch.from_numpy(X), torch.from_numpy(y)

    def __len__(self):
        return len(self.rolloffs)*self.n_batch


def get_loaders(split_percent=0.8):
    dataset = SignalDataset()
    train_size = int(split_percent * len(dataset))
    test_size = len(dataset) - train_size

    return random_split(dataset, [train_size, test_size])
