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

class DataReconstructionDataset(Dataset):
    def __init__(self, path, mode="train", id=1) -> None:
        self.path = path
        self.train_path = os.path.join(self.path, "train")
        self.test_path = os.path.join(self.path, "test")
        self.mode = mode

        if id == 1:
            self.mask = (4,)
        else:
            self.mask = (2, 3, 4)

        self.min_max = pd.read_csv(os.path.join(self.path, "min_max.csv")).values
        
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

            for signal_name in sorted(os.listdir(self.train_path)):
                name, _, _ = scipy.io.whosmat(os.path.join(self.path, "train", signal_name))[0]
                x = scipy.io.loadmat(os.path.join(self.path, "train", signal_name))[name]
                x = min_max_scaler(x, self.min_max)
                

                sliding_signal = sliding_window(x, window=1024, stride=1024)

                train_x, valid_x = train_test_split(sliding_signal, train_size=0.7, random_state=0)
                valid_x, test_x = train_test_split(valid_x, train_size=(2/3.), random_state=0)
                


                self.train_data += train_x
                self.valid_data += valid_x
                self.test_data += test_x
            
            self.train_mask = create_mask_array(len(self.train_data))
            self.valid_mask = create_mask_array(len(self.valid_data))
            self.test_mask = create_mask_array(len(self.test_data))
                

        else:
            self.data_path = os.path.join(self.path, "test", f"data_noised_testset{id}.mat")
            self.name, _, _ = scipy.io.whosmat(self.data_path)[0]
            self.signal = scipy.io.loadmat(self.data_path)[self.name]
            self.signal = min_max_scaler(self.signal, self.min_max)
            self.evaluate_data = sliding_window(self.signal, 1024, 1024)
            

            




    def __len__(self) -> int:
        if self.mode == "train" :
            return len(self.train_data)
        
        elif self.mode == "valid":
            return len(self.valid_data)

        elif self.mode == "test":
            return len(self.test_data)
        
        else:
            return len(self.evaluate_data)

    def __getitem__(self, index) -> torch.tensor:
        if self.mode == "train":
            mask = self.train_mask[index]
            input = self.train_data[index]
            target = np.copy(input)
            for i in range(5):
                if i in mask:
                    input[i, :] = input[i, :] * 0

            input = torch.tensor(input, dtype=torch.float32)
            target = torch.tensor(target, dtype=torch.float32)


            return input, target

        elif self.mode == "valid":
            mask = self.valid_mask[index]
            input = self.valid_data[index]
            target = np.copy(input)
            for i in range(5):
                if i in mask:
                    input[i, :] = input[i, :] * 0

            input = torch.tensor(input, dtype=torch.float32)
            target = torch.tensor(target, dtype=torch.float32)

            return input, target
        
        elif self.mode == "test":
            mask = self.test_mask[index]
            input = self.test_data[index]
            target = np.copy(input)
            for i in range(5):
                if i in mask:
                    input[i, :] = input[i, :] * 0

            input = torch.tensor(input, dtype=torch.float32)
            target = torch.tensor(target, dtype=torch.float32)

            return input, target

        else:
            input = self.evaluate_data[index]
            for i in range(5):
                if i in self.mask:
                    input[i, :] = input[i, :] * 0
            input = torch.tensor(input, dtype=torch.float32)
            return input 
    


if __name__ == "__main__":
    source = "Accleration"
    dataset = DataReconstructionDataset(path=f"./Data/{source}", mode="evaluate", id=1)
    # print(dataset.__len__())
    dataloader = DataLoader(dataset, batch_size=32)
    for batch in dataloader:
        if len(batch) == 2:
            print(batch[0])
            print(batch[0].shape)
            print(batch[1])
            print(batch[1].shape)
        else:
            print(batch)
            print(batch.shape)
        
