import os
import torch
import scipy
import random
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

def sliding_window(signal, window, stride):
    length = signal.shape[-1]
    i = 0
    x = []
    while i < length-window:
        x.append(signal[:, i:i+window])
        i += stride
    return x

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
    def __init__(self, path, data_type="1D",mode="train", task="classification") -> None:
        self.path = path
        self.train_path = os.path.join(self.path, "Displacement", "train")
        self.mode = mode
        self.task = task
        self.data_type = data_type

        self.min_max = pd.read_csv(os.path.join(self.path, "Displacement", "min_max.csv")).values
        label_file = pd.read_csv(os.path.join(self.path, "Displacement", "label.csv"))
        
        self.data = []
        self.labels = []
        self.damage_state = []

        for situation, signal_name in enumerate(sorted(os.listdir(self.train_path))):
            name, _, _ = scipy.io.whosmat(os.path.join(self.path, "Displacement", "train", signal_name))[0]
            x = scipy.io.loadmat(os.path.join(self.path, "Displacement", "train", signal_name))[name]
            x = min_max_scaler(x, self.min_max)

            sliding_signal = sliding_window(x, window=1024, stride=1024)
            self.data += sliding_signal

            damage = label_file.loc[label_file['File Name'] == signal_name].values[0, 1:4].astype(float)
            if task == "classification":
                damage = one_hot(damage)
            
            self.damage_state += [damage for i in range(len(sliding_signal))]
            self.labels += [situation+1 for i in range(len(sliding_signal))]



        self.index = np.array([i for i in range(len(self.data))])
        self.damage_state = np.array(self.damage_state)
        self.labels = np.array(self.labels)
        self.data= np.array(self.data)
            






    def __len__(self) -> int:
        return len(self.data)
        # return 512

    def __getitem__(self, index) -> torch.tensor:
        input = self.data[index]
        label = self.labels[index]
        damage_state = self.damage_state[index]

        input = torch.tensor(input, dtype=torch.float32)
        damage_state = torch.tensor(damage_state, dtype=torch.float32)
        return input, damage_state, label


if __name__ == "__main__":
    dataset = DamageDataGenerationDataset(path="./Data")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    a = next(dataloader.__iter__())
    print(a)
