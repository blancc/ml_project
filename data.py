# Dataset classes
from torch.utils.data import Dataset, DataLoader, random_split


class SignalDataset(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass


def get_loaders(split_percent=0.8):
    dataset = SignalDataset()
    train_size = int(split_percent * len(dataset))
    test_size = len(dataset) - train_size

    return random_split(dataset, [train_size, test_size])
