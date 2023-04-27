import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader


def min_max_scaler(signal, min_max):
    new_signal = np.copy(signal)
    for i in range(signal.shape[0]):
        new_signal[i, :] = (signal[i, :] - min_max[i][0])/(min_max[i][1] - min_max[i][0])
    return new_signal


class FeatureExtractionDataset(Dataset):
    def __init__(self, path, source, mode="train") -> None:
        self.root = os.path.join(path, source)
        self.mode = mode
        self.path = os.path.join(self.root, self.mode)
        self.files = os.listdir(self.path)

        self.min_max = pd.read_csv(os.path.join(self.root, "min_max.csv")).values


    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx) -> torch.tensor:
        input = np.load(os.path.join(self.path, self.files[idx]))
        input = min_max_scaler(input, self.min_max)
        return torch.tensor(input, dtype=torch.float32)

            
            



if __name__ == "__main__":
    dataset = FeatureExtractionDataset(path="./Data", source="Displacement", mode="test")
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(next(iter(dataloader))[-2])