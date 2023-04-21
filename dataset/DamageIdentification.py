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

class DamageIdentificationDataset(Dataset):
    def __init__(self, path, data_type="1D", mode="train", task="classification") -> None:
        self.path = path
        self.train_path = os.path.join(self.path, "train")
        self.test_path = os.path.join(self.path, "test")
        self.mode = mode
        self.task = task
        self.data_type = data_type
        self.scaler = MinMaxScaler()
        label_file = pd.read_csv(os.path.join(self.path, "label.csv"))

        

        if mode != "test":
            self.train_data = []
            self.train_label = []
            self.train_id = []

            self.valid_data = []
            self.valid_label = []
            self.valid_id = []

            for situation, signal_name in enumerate(sorted(os.listdir(self.train_path))):
                name, _, _ = scipy.io.whosmat(os.path.join(self.path, "train", signal_name))[0]
                x = scipy.io.loadmat(os.path.join(self.path, "train", signal_name))[name]
                x = self.scaler.fit_transform(x)
                

                sliding_signal = sliding_window(x, window=1024, stride=1024)
                label = label_file.loc[label_file['File Name'] == signal_name].values[0, 1:4].astype(float)
                if task == "classification":
                    label = one_hot(label)
    
                label = [label for i in range(len(sliding_signal))]
                situation = [situation + 1 for i in range(len(sliding_signal))]
                split = int(len(sliding_signal) * 0.8)
                self.train_data += sliding_signal[:split]
                self.train_label += label[:split]
                self.train_id += situation[:split]

                self.valid_data += sliding_signal[split:]
                self.valid_label += label[split:]
                self.valid_id += situation[split:]


        if mode == "test":
            self.test_data = []
            self.signal_id = []
            for id, signal_name in enumerate(sorted(os.listdir(self.test_path))):
                name, _, _ = scipy.io.whosmat(os.path.join(self.path, "test", signal_name))[0]
                x = scipy.io.loadmat(os.path.join(self.path, "test", signal_name))[name]
                x = self.scaler.fit_transform(x)
                

                sliding_signal = sliding_window(x, window=1024, stride=1024)
                self.test_data += sliding_signal
                self.signal_id += [id for i in range(len(sliding_signal))]





    def __len__(self) -> int:
        if self.mode == "train":
            return len(self.train_data)
        elif self.mode == "valid":
            return len(self.valid_data)
        else:
            return len(self.test_data)

    def __getitem__(self, idx) -> torch.tensor:
        
        if self.mode == "train":
            input = torch.tensor(self.train_data[idx], dtype=torch.float32)
            signal_id = torch.tensor(self.train_id[idx], dtype=torch.float32)
            if self.data_type == "2D":
                input = torch.unsqueeze(input, axis=1)

            if self.task == "classification":
                label1 = torch.tensor(self.train_label[idx][0], dtype=torch.float32)
                label2 = torch.tensor(self.train_label[idx][1], dtype=torch.float32)
                label3 = torch.tensor(self.train_label[idx][2], dtype=torch.float32)
                return input, label1, label2, label3, signal_id
            else:
                label = torch.tensor(self.train_label[idx], dtype=torch.float32)
                return input, label, signal_id
            
        
        elif self.mode == "valid":
            input = torch.tensor(self.valid_data[idx], dtype=torch.float32)
            signal_id = torch.tensor(self.valid_id[idx], dtype=torch.float32)
            if self.data_type == "2D":
                input = torch.unsqueeze(input, axis=1)

            if self.task == "classification":
                label1 = torch.tensor(self.valid_label[idx][0], dtype=torch.float32)
                label2 = torch.tensor(self.valid_label[idx][1], dtype=torch.float32)
                label3 = torch.tensor(self.valid_label[idx][2], dtype=torch.float32)
                return input, label1, label2, label3, signal_id
            else:
                label = torch.tensor(self.valid_label[idx], dtype=torch.float32)
                return input, label, signal_id
        else:
            input = torch.tensor(self.test_data[idx], dtype=torch.float32)
            signal_id = torch.tensor(self.signal_id[idx], dtype=torch.float32)
            if self.data_type == "2D":
                input = torch.unsqueeze(input, axis=1)
            return input, signal_id


if __name__ == "__main__":
    dataset = DamageIdentificationDataset(path="./Data", mode="train")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    for i in dataloader:
        print(i[1].shape)
