from nlb_tools.nwb_interface import NWBDataset
from sparsebmi.data_loading.dataset import create_datasets_with_history
import pandas as pd
import numpy as np


def load_nlb(nlb_dataset_path, binsize_ms, seq_len, batch_size, train_val_test=(0.7, 0.1, 0.2), pred_type='pv',
             normalize_x=True, normalize_y=True):
    # load dataset
    dataset = NWBDataset(nlb_dataset_path, "*train", split_heldout=False)

    # bin and get data
    dataset.resample(binsize_ms)
    if pred_type == 'pv':
        dataset.movement = pd.DataFrame({
            'finger_pos_x': dataset.data['finger_pos']['x'],
            'finger_pos_y': dataset.data['finger_pos']['y'],
            'finger_vel_x': dataset.data['finger_vel']['x'],
            'finger_vel_y': dataset.data['finger_vel']['y'],
        })
    elif pred_type == 'p':
        dataset.movement = pd.DataFrame({
            'finger_pos_x': dataset.data['finger_pos']['x'],
            'finger_pos_y': dataset.data['finger_pos']['y'],
        })
    elif pred_type == 'v':
        dataset.movement = pd.DataFrame({
            'finger_vel_x': dataset.data['finger_vel']['x'],
            'finger_vel_y': dataset.data['finger_vel']['y'],
        })
    else:
        raise ValueError("pred_type must be one of ['pv', 'p', 'v']")
    x = dataset.data['spikes'].astype(np.float32).to_numpy()
    y = dataset.movement.to_numpy()

    # remove any bins with NaNs
    nan_idx = np.isnan(x).any(axis=1)
    x = x[~nan_idx]
    y = y[~nan_idx]

    # split into train, val, test
    train_idx = int(len(x) * train_val_test[0])
    val_idx = int(len(x) * (train_val_test[0] + train_val_test[1]))
    test_idx = int(len(x) * (train_val_test[0] + train_val_test[1] + train_val_test[2]))
    x_train, y_train = x[:train_idx], y[:train_idx]
    x_val, y_val = x[train_idx:val_idx], y[train_idx:val_idx]
    x_test, y_test = x[val_idx:test_idx], y[val_idx:test_idx]

    # normalize
    if normalize_x:
        x_train_mean = np.mean(x_train, axis=0)
        x_train_std = np.std(x_train, axis=0)
        x_train = (x_train - x_train_mean) / (x_train_std + 1e-6)
        x_val = (x_val - x_train_mean) / (x_train_std + 1e-6)
        x_test = (x_test - x_train_mean) / (x_train_std + 1e-6)
    if normalize_y:
        y_train_mean = np.mean(y_train, axis=0)
        y_train_std = np.std(y_train, axis=0)
        y_train = (y_train - y_train_mean) / (y_train_std + 1e-6)
        y_val = (y_val - y_train_mean) / (y_train_std + 1e-6)
        y_test = (y_test - y_train_mean) / (y_train_std + 1e-6)

    print("Train:", x_train.shape, y_train.shape)
    print("Val:", x_val.shape, y_val.shape)
    print("Test:", x_test.shape, y_test.shape)

    # add history and create dataloaders
    loader_train, loader_val, loader_test = create_datasets_with_history(x_train, y_train, x_val, y_val, x_test, y_test,
                                                                         batch_size=batch_size, seq_len=seq_len)

    return loader_train, loader_val, loader_test
