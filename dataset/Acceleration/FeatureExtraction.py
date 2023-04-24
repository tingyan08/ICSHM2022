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
    def __init__(self, path, mode="train", triplet=False, Damage=False) -> None:
        self.path = os.path.join(path, "Acceleration")
        self.triplet = triplet
        self.Damage = Damage
        self.train_path = os.path.join(self.path, "train")
        self.test_path = os.path.join(self.path, "test")
        self.mode = mode

        self.min_max = pd.read_csv(os.path.join(self.path, "min_max.csv")).values
        
        
        if self.mode != "evaluate":

            self.train_data = []
            self.valid_data = []
            self.test_data = []

            for situation, signal_name in enumerate(sorted(os.listdir(self.train_path))):
                name, _, _ = scipy.io.whosmat(os.path.join(self.path, "train", signal_name))[0]
                x = scipy.io.loadmat(os.path.join(self.path, "train", signal_name))[name]
                x = min_max_scaler(x, self.min_max)
                

                sliding_signal = sliding_window(x, window=1024, stride=1024)

                train_x, valid_x = train_test_split(sliding_signal, train_size=0.7, random_state=0)
                valid_x, test_x = train_test_split(valid_x, train_size=(2/3.), random_state=0)

                self.train_data += train_x
                self.valid_data += valid_x
                self.test_data += test_x
            

        elif self.mode == "evaluate":
            self.evaluation_data = []

            for situation, signal_name in enumerate(sorted(os.listdir(self.test_path))):
                name, _, _ = scipy.io.whosmat(os.path.join(self.path, "test", signal_name))[0]
                x = scipy.io.loadmat(os.path.join(self.path, "test", signal_name))[name]
                x = min_max_scaler(x, self.min_max)
                
                sliding_signal = sliding_window(x, window=1024, stride=1024)
                self.evaluation_data += sliding_signal

        else:
            raise Exception("Invalid mode")



    def __len__(self) -> int:
        if self.mode == "train":
            return len(self.train_data)
        elif self.mode == "valid":
            return len(self.valid_data)
        elif self.mode == "test":
            return len(self.test_data)
        elif self.mode == "evaluate":
            return len(self.evaluate_data)
        else:
            raise Exception("Invalid mode")

    def __getitem__(self, idx) -> torch.tensor:

        if self.mode == "train":

            anchor_signal = self.train_data[idx]
            anchor_signal = torch.tensor(anchor_signal, dtype=torch.float32)

            return anchor_signal
        
        elif self.mode == "valid":
            anchor_signal = self.valid_data[idx]
            anchor_signal = torch.tensor(anchor_signal, dtype=torch.float32)

            return anchor_signal
        
        elif self.mode == "test":
            anchor_signal = self.test_data[idx]
            anchor_signal = torch.tensor(anchor_signal, dtype=torch.float32)

            return anchor_signal

            
            





if __name__ == "__main__":
    dataset = FeatureExtractionDataset(path="./Data", mode="valid")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(next(iter(dataloader))[-2])