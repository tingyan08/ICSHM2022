import os
import torch
import scipy
import random
import torch.nn as nn
import pandas as pd
import numpy as np
from itertools import combinations
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def sliding_window(signal, window, stride):
    length = signal.shape[-1]
    i = 0
    x = []
    while i < length-window:
        x.append(signal[:, i:i+window])
        i += stride
    return x

def min_max_scaler(signal, min_max):
    for i in range(signal.shape[0]):
        signal[i, :] = (signal[i, :] - min_max[i][1])/(min_max[i][0] - min_max[i][1])
    return signal

class DataDrivenDatasetTask(Dataset):
    def __init__(self, path, mode="train", task="A", window=512, stride=431, data_type="1D") -> None:
        self.path = path
        self.task = task
        self.mode = mode
        self.data_type = data_type

        self.window = window
        self.stride = stride

        if self.task == "A":
            self.mask = np.array([True, True, True, True, False])
            self.output_ch = 1
        else:
            self.mask = np.array([True, True, False, False, False])
            self.output_ch = 3

        self.min_max = pd.read_csv(os.path.join(self.path, "min_max.csv")).values
        if self.mode != "test":
            self.data_path = os.path.join(self.path, "data_noised.mat")
            self.name, _, _ = scipy.io.whosmat(self.data_path)[0]
            self.signal = scipy.io.loadmat(self.data_path)[self.name]
            self.signal = min_max_scaler(self.signal, self.min_max)
            data = sliding_window(self.signal, self.window, self.stride)
            self.train_data, self.valid_data = train_test_split(data, test_size=0.2, random_state=0)

        else:
            if task == "A":
                self.data_path = os.path.join(self.path, "data_noised_testset.mat")
            else:
                self.data_path = os.path.join(self.path, "data_noised_testset2.mat")

            self.name, _, _ = scipy.io.whosmat(self.data_path)[0]
            self.signal = scipy.io.loadmat(self.data_path)[self.name]



    def __len__(self) -> int:
        if self.mode == "train" :
            return len(self.train_data)
        
        elif self.mode == "validation":
            return len(self.valid_data)
        
        else:
            return 1

    def __getitem__(self, index) -> torch.tensor:
        if self.mode == "train":
            signal = self.train_data[index]
            input = torch.tensor(signal[self.mask, :], dtype=torch.float32)
            target = torch.tensor(signal[~self.mask, :], dtype=torch.float32)
            
            if self.data_type == "2D":
                input = input.unsqueeze(0)
                target = target.unsqueeze(0)
            return input, target

        elif self.mode == "validation":
            signal = self.valid_data[index]
            input = torch.tensor(signal[self.mask, :], dtype=torch.float32)
            target = torch.tensor(signal[~self.mask, :], dtype=torch.float32)

            if self.data_type == "2D":
                input = input.unsqueeze(0)
                target = target.unsqueeze(0)

            return input, target

        else:
            x = np.copy(self.signal)

            for i, mask in enumerate(self.mask):
                x[i, :] = (x[i, :] - self.min_max[i, 0]) / (self.min_max[i, 1] - self.min_max[i, 0]) 
                x[i, :] = x[i, :] * mask

            input = torch.tensor(x[self.mask, :], dtype=torch.float32)

            if self.data_type == "2D":
                input = input.unsqueeze(0)

            return input
    


if __name__ == "__main__":
    dataset = DataDrivenDatasetTask(path="./Data", mode="train", task="B", data_type="2D")
    # print(dataset.__len__())
    dataloader = DataLoader(dataset, batch_size=30)
    data = next(dataloader.__iter__())
    print(data[0].shape)
    print(data[1].shape)
