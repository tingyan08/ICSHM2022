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

class DamageIdentificationDataset(Dataset):
    def __init__(self, path, mode="train", classification=False) -> None:
        self.path = os.path.join(path, "Displacement")
        self.train_path = os.path.join(self.path, "train")
        self.test_path = os.path.join(self.path, "test")
        self.mode = mode
        self.classification = classification

        self.min_max = pd.read_csv(os.path.join(self.path, "min_max.csv")).values
        label_file = pd.read_csv(os.path.join(self.path, "label.csv"))

        if self.mode != "evaluate":
        
            self.train_data = []
            self.train_label = []
            self.train_id = []

            self.valid_data = []
            self.valid_label = []
            self.valid_id = []

            self.test_data = []
            self.test_label = []
            self.test_id = []

            for situation, signal_name in enumerate(sorted(os.listdir(self.train_path))):
                name, _, _ = scipy.io.whosmat(os.path.join(self.path, "train", signal_name))[0]
                x = scipy.io.loadmat(os.path.join(self.path, "train", signal_name))[name]
                x = min_max_scaler(x, self.min_max)
                length = x.shape[1]
                label = label_file.loc[label_file['File Name'] == signal_name].values[0, 1:4].astype(float)

                crop_range = [0, int(0.7 * length), int(0.9 * length), int(length)]
                train_x = sliding_window(x[:, crop_range[0]:crop_range[1]], window=1024, stride=128)
                valid_x = sliding_window(x[:, crop_range[1]:crop_range[2]], window=1024, stride=128)
                test_x = sliding_window(x[:, crop_range[2]:crop_range[3]], window=1024, stride=128)
                
                if self.classification:
                    label = one_hot(label)


                self.train_data += train_x
                self.train_label += [label for i in range(len(train_x))]
                self.train_id += [situation + 1 for i in range(len(train_x))]

                self.valid_data += valid_x
                self.valid_label += [label for i in range(len(valid_x))]
                self.valid_id += [situation + 1 for i in range(len(valid_x))]

                self.test_data += test_x
                self.test_label += [label for i in range(len(test_x))]
                self.test_id += [situation + 1 for i in range(len(test_x))]


        else:
            self.evaluation_data = []

            for situation, signal_name in enumerate(sorted(os.listdir(self.test_path))):
                name, _, _ = scipy.io.whosmat(os.path.join(self.path, "test", signal_name))[0]
                x = scipy.io.loadmat(os.path.join(self.path, "test", signal_name))[name]
                x = min_max_scaler(x, self.min_max)
                
                sliding_signal = sliding_window(x, window=1024, stride=1024)
                self.evaluation_data += sliding_signal




    def __len__(self) -> int:
        if self.mode == "train":
            return len(self.train_data)
        elif self.mode == "valid":
            return len(self.valid_data)
        elif self.mode == "test":
            return len(self.test_data)
        else:
            return len(self.evaluation_data)

    def __getitem__(self, idx) -> torch.tensor:
        
        if self.mode == "train":
            input = torch.tensor(self.train_data[idx], dtype=torch.float32)
            signal_id = torch.tensor(self.train_id[idx], dtype=torch.float32)
            label = torch.tensor(self.train_label[idx], dtype=torch.float32)
            if self.classification:
                return input, label[0], label[1], label[2], signal_id
            else:
                return input, label, signal_id

            
        
        elif self.mode == "valid":
            input = torch.tensor(self.valid_data[idx], dtype=torch.float32)
            signal_id = torch.tensor(self.valid_id[idx], dtype=torch.float32)
            label = torch.tensor(self.valid_label[idx], dtype=torch.float32)
            if self.classification:
                return input, label[0], label[1], label[2], signal_id
            else:
                return input, label, signal_id
        
        elif self.mode == "test":
            input = torch.tensor(self.test_data[idx], dtype=torch.float32)
            signal_id = torch.tensor(self.test_id[idx], dtype=torch.float32)
            label = torch.tensor(self.test_label[idx], dtype=torch.float32)
            if self.classification:
                return input, label[0], label[1], label[2], signal_id
            else:
                return input, label, signal_id

        else:
            input = torch.tensor(self.evaluation_data[idx], dtype=torch.float32)

            return input


if __name__ == "__main__":
    dataset = DamageIdentificationDataset(path=f"./Data/", mode="train")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    for i in dataloader:
        print(i[1].shape)
