import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader


def min_max_scaler(signal, min_max):
    new_signal = np.copy(signal)
    for i in range(signal.shape[0]):
        new_signal[i, :] = (signal[i, :] - min_max[i][0])/(min_max[i][1] - min_max[i][0])
    return new_signal

class DamageIdentificationDataset(Dataset):
    def __init__(self, path, source, mode="train") -> None:
        self.root = os.path.join(path, source)
        self.mode = mode
        self.path = os.path.join(self.root, self.mode)
        self.min_max = pd.read_csv(os.path.join(self.root, "min_max.csv")).values
        self.label_files = pd.read_csv(os.path.join(self.root, f"{mode}.csv"))
        self.files = self.label_files.loc[:, "name"].values
        self.target = self.label_files.loc[:, "label1":"label3"].values



    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx) -> torch.tensor:
        input = np.load(os.path.join(self.path, self.files[idx]))
        input = min_max_scaler(input, self.min_max)
        target = self.target[idx]
        return torch.tensor(input, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)


if __name__ == "__main__":
    dataset = DamageIdentificationDataset(path=f"./Data/", source="Displacement", mode="train")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    for i in dataloader:
        print(i)
