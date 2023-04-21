import os
import torch
import scipy
import random
import torch.nn as nn
import pandas as pd
import numpy as np
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
    new_signal = np.copy(signal)
    for i in range(signal.shape[0]):
        new_signal[i, :] = (signal[i, :] - min_max[i][0])/(min_max[i][1] - min_max[i][0])
    return new_signal

def one_hot(label):
    no7 = np.zeros(6)
    no22 = np.zeros(6)
    no38 = np.zeros(6)
    # No.7
    no7[int(label[0]*10)] = 1
    # No.22
    no22[int(label[1]*10)] = 1
    # No.38
    no38[int(label[2]*10)] = 1

    return no7, no22, no38


class FeatureExtractionDataset(Dataset):
    def __init__(self, path, data_type="1D", mode="train", task="classification") -> None:
        self.path = path
        self.train_path = os.path.join(self.path, "train")
        self.mode = mode
        self.task = task
        self.data_type = data_type

        self.min_max = pd.read_csv(os.path.join(self.path, "min_max.csv")).values
        label_file = pd.read_csv(os.path.join(self.path, "label.csv"))
        
        self.data = []
        self.labels = []
        self.damage_state = []

        for situation, signal_name in enumerate(sorted(os.listdir(self.train_path))):
            name, _, _ = scipy.io.whosmat(os.path.join(self.path, "train", signal_name))[0]
            x = scipy.io.loadmat(os.path.join(self.path, "train", signal_name))[name]
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

    def __getitem__(self, idx) -> torch.tensor:

        anchor_label = self.labels[idx]
        anchor_signal = self.data[idx]
        anchor_state = self.damage_state[idx]

        positive_list = self.index[self.index!=idx][self.labels[self.index!=idx]==anchor_label]
        positive_item = random.choice(positive_list)
        positive_signal = self.data[positive_item]
        positive_state = self.damage_state[positive_item]

        negative_list = self.index[self.index!=idx][self.labels[self.index!=idx]!=anchor_label]
        negative_item = random.choice(negative_list)
        negative_signal = self.data[negative_item]
        negative_state = self.damage_state[negative_item]

        anchor_signal = torch.tensor(anchor_signal, dtype=torch.float32)
        positive_signal = torch.tensor(positive_signal, dtype=torch.float32)
        negative_signal = torch.tensor(negative_signal, dtype=torch.float32)



        return (anchor_signal, positive_signal, negative_signal), \
            (anchor_state, positive_state, negative_state), anchor_label



if __name__ == "__main__":
    dataset = FeatureExtractionDataset(path="./Data", mode="train")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(next(iter(dataloader))[-2])