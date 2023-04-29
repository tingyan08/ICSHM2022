import os
import torch
import scipy
import random
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader


def min_max_scaler(signal, min_max):
    new_signal = np.copy(signal)
    for i in range(signal.shape[0]):
        new_signal[i, :] = (signal[i, :] - min_max[i][0])/(min_max[i][1] - min_max[i][0])
    return new_signal

def one_hot(label):
    one_hot_label = np.zeros(18, dtype=float)
    # No.7
    one_hot_label[int(label[0]*10)] = 1
    # No.22
    one_hot_label[int(label[1]*10)+6] = 1
    # No.38
    one_hot_label[int(label[2]*10)+12] = 1

    return one_hot_label


class DamageDataGenerationDataset(Dataset):
    def __init__(self, path, source) -> None:
        self.root = os.path.join(path, source)
        self.min_max = pd.read_csv(os.path.join(self.root, "min_max.csv")).values

        self.files = []
        self.labels = []
        
        self.train_label_file = pd.read_csv(os.path.join(self.root, "train.csv"))
        train_files = self.train_label_file.loc[:, "name"].values
        train_labels = self.train_label_file.loc[:, "label1":"label3"].values
        self.files += ([os.path.join(self.root, "train", i) for i in train_files])
        self.labels += ([i for i in train_labels])

        self.valid_label_file = pd.read_csv(os.path.join(self.root, "valid.csv"))
        valid_files = self.valid_label_file.loc[:, "name"].values
        valid_labels = self.valid_label_file.loc[:, "label1":"label3"].values
        self.files += ([os.path.join(self.root, "valid", i) for i in valid_files])
        self.labels += ([i for i in valid_labels])
        
        self.test_label_file = pd.read_csv(os.path.join(self.root, "test.csv"))
        test_files = self.test_label_file.loc[:, "name"].values
        test_labels = self.test_label_file.loc[:, "label1":"label3"].values
        self.files += ([os.path.join(self.root, "test", i) for i in test_files])
        self.labels += ([i for i in test_labels])


        
            






    def __len__(self) -> int:
        return len(self.files)
        # return 512

    def __getitem__(self, index) -> torch.tensor:
        input = np.load(self.files[index])
        input = min_max_scaler(input, self.min_max)
        label = one_hot(self.labels[index])

        input = torch.tensor(input, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        return input, label


if __name__ == "__main__":
    dataset = DamageDataGenerationDataset(path="./Data")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    a = next(dataloader.__iter__())
    print(a)
