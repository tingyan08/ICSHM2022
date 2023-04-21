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


def create_mask_array(total_n):
    mask_list = []
    for num_mask in range(1, 5):
        for id_list in combinations(range(5), num_mask):
            mask_list.append(id_list)

    while len(mask_list) < total_n:
        mask_list += mask_list

    mask_list = mask_list[:total_n]
    return mask_list

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

class DataDrivenDataset(Dataset):
    def __init__(self, path, mode="train", task="A", window=1024, stride=1024, data_type="1D") -> None:
        self.path = path
        self.task = task
        self.mode = mode
        self.data_type = data_type

        self.window = window
        self.stride = stride
        self.output_ch = 5
        self.min_max = pd.read_csv(os.path.join(self.path, "min_max.csv")).values
        
        data = []
        self.data_id = []
        for id, filename in enumerate(os.listdir(os.path.join(self.path, "train"))):
            self.data_path = os.path.join(self.path, "train", filename)
            self.name, _, _ = scipy.io.whosmat(self.data_path)[0]
            self.signal = scipy.io.loadmat(self.data_path)[self.name]
            self.signal = min_max_scaler(self.signal, self.min_max)
            temp = sliding_window(self.signal, self.window, self.stride)
            data += temp
            self.data_id += [id+1 for i in range(len(temp))]
        
        self.train_data, self.valid_data = train_test_split(data, test_size=0.2, random_state=0)
        if task == "A":
            self.train_mask = [(4,) for i in range(len(self.train_data))]
            self.valid_mask = [(4,) for i in range(len(self.valid_data))]
        elif task == "B":
            self.train_mask = [(2,3,4) for i in range(len(self.train_data))]
            self.valid_mask = [(2,3,4) for i in range(len(self.valid_data))]
        else:
            self.train_mask, self.valid_mask = create_mask_array(len(self.train_data)), create_mask_array(len(self.valid_data))


            




    def __len__(self) -> int:
        if self.mode == "train" :
            return len(self.train_data)
        
        elif self.mode == "valid":
            return len(self.valid_data)
        
        else:
            return 1

    def __getitem__(self, index) -> torch.tensor:
        if self.mode == "train":
            data = self.train_data[index]

            if self.task == "A":
                input = data[:4, :]
                target = data[4:, :]
            elif self.task == "B":
                input = data[:2, :]
                target = data[2:5, :]
            else:
                mask = self.train_mask[index]
                target = np.copy(data)
                input = np.copy(data)
                for i in range(5):
                    if i in mask:
                        input[i, :] = input[i, :] * 0 + 0.5

            

            
            input = torch.tensor(input, dtype=torch.float32)
            target = torch.tensor(target, dtype=torch.float32)

            if self.data_type == "2D":
                input = input.unsqueeze(0)
                target = target.unsqueeze(0)

            return input, target

        elif self.mode == "valid":
            data = self.valid_data[index]

            if self.task == "A":
                input = data[:4, :]
                target = data[4:, :]
            elif self.task == "B":
                input = data[:2, :]
                target = data[2:5, :]
            else:
                mask = self.valid_mask[index]
                target = np.copy(data)
                input = np.copy(data)
                for i in range(5):
                    if i in mask:
                        input[i, :] = input[i, :] * 0 + 0.5

            input = torch.tensor(input, dtype=torch.float32)
            target = torch.tensor(target, dtype=torch.float32)

            if self.data_type == "2D":
                input = input.unsqueeze(0)
                target = target.unsqueeze(0)

            return input, target
    


if __name__ == "__main__":
    dataset = DataDrivenDataset(path="./Data", mode="train", task="A")
    # print(dataset.__len__())
    dataloader = DataLoader(dataset, batch_size=32)
    data = next(dataloader.__iter__())
    print(data)
