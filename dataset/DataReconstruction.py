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

def min_max_scaler(signal, min_max):
    new_signal = np.copy(signal)
    for i in range(signal.shape[0]):
        new_signal[i, :] = (signal[i, :] - min_max[i][0])/(min_max[i][1] - min_max[i][0])
    return new_signal

class DataReconstructionDataset(Dataset):
    def __init__(self, path, source, mode="train") -> None:
        self.root = os.path.join(path, source)
        self.mode = mode
        self.path = os.path.join(self.root, self.mode)
        self.files = os.listdir(self.path)

        self.min_max = pd.read_csv(os.path.join(self.root, "min_max.csv")).values
        
            
        self.mask = create_mask_array(len(self.files))
                
            

            




    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx) -> torch.tensor:
        mask = self.mask[idx]
        target = np.load(os.path.join(self.path, self.files[idx]))
        target = min_max_scaler(target, self.min_max)
        masked_input = np.zeros_like(target)
        for i in range(5):
            if i in mask:
                masked_input[i, :] = target[i, :] * 0
            else:
                masked_input[i, :] = target[i, :]

        masked_input = torch.tensor(masked_input, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)


        return masked_input, target

        
    


if __name__ == "__main__":
    dataset = DataReconstructionDataset(path=f"./Data", source="Displacement", mode="train")
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
        
