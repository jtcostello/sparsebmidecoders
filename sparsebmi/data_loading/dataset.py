import torch
from torch.utils.data import Dataset, DataLoader


def create_datasets_with_history(x_train, y_train, x_val, y_val, x_test, y_test, batch_size=64, seq_len=3,
                                 mem_optimized=False):
    """ Adds history and create dataloaders for train/val/test """

    # create datasets
    if mem_optimized:
        # dataset that adds history when requested
        dataset_train = SequenceDatasetAddHist(torch.from_numpy(x_train), torch.from_numpy(y_train), x_seq_len=seq_len)
        dataset_val = SequenceDatasetAddHist(torch.from_numpy(x_val), torch.from_numpy(y_val), x_seq_len=seq_len)
        dataset_test = SequenceDatasetAddHist(torch.from_numpy(x_test), torch.from_numpy(y_test), x_seq_len=seq_len)
    else:
        # add time history to input features
        x_train = add_time_history(x_train, seq_len=seq_len)
        x_val = add_time_history(x_val, seq_len=seq_len)
        x_test = add_time_history(x_test, seq_len=seq_len)

        # create datasets
        dataset_train = SequenceDataset(x_train, torch.from_numpy(y_train))
        dataset_val = SequenceDataset(x_val, torch.from_numpy(y_val))
        dataset_test = SequenceDataset(x_test, torch.from_numpy(y_test))

    # create dataloaders
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=True)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    return loader_train, loader_val, loader_test


def add_time_history(x, seq_len=3):
    """
    Adds time history to the input features
    Input shape (num_samples, num_chans)
    Output shape (num_samples, seq_len, num_chans)
    """
    xin = torch.tensor(x)

    # add time delays to input features
    xhist = torch.zeros((int(xin.shape[0]), int(xin.shape[1]), seq_len))
    xhist[:, :, 0] = xin
    for i in range(1, seq_len):
        xhist[i:, :, i] = xin[0:-i, :]

    # make the last timestep the most recent data
    xhist = torch.flip(xhist, (2,))

    return xhist


class SequenceDataset(Dataset):
    """Simple dataset for sequences of continuous data"""
    def __init__(self, x, y):
        self.x = x.to(torch.float)
        self.y = y.to(torch.float)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx, :], self.y[idx]


class SequenceDatasetAddHist(Dataset):
    """Dataset for sequences of continuous data that adds history when requested (to save memory)"""
    def __init__(self, x, y, x_seq_len=3):
        self.x = x.to(torch.float)
        self.y = y.to(torch.float)
        self.x_seq_len = x_seq_len

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # idx is the index of the last timestep in the sequence
        if idx >= (self.x_seq_len - 1):
            x_out = self.x[idx - self.x_seq_len + 1:idx + 1, :]
        else:
            # if idx is less than the sequence length, pad with zeros
            x_out = torch.zeros((self.x_seq_len, self.x.shape[1]))
            x_out[-idx - 1:, :] = self.x[:idx + 1, :]

        # reshape from (seq_len, chans) to (chans, seq_len)
        x_out = x_out.permute(1, 0)

        return x_out, self.y[idx]
