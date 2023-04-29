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
    def __init__(self, path, source, mode="train", classification=False) -> None:
        self.classification = classification

        self.files = []
        self.target = []

        for s in source:
            root = os.path.join(path, s)
            dir = os.path.join(root, mode)
            label_files = pd.read_csv(os.path.join(root, f"{mode}.csv"))
            files = label_files.loc[:, "name"].values
            self.files += [os.path.join(dir, i) for i in files]

            target = label_files.loc[:, "label1":"label3"].values
            self.target += [i for i in target]

        self.min_max = pd.read_csv(os.path.join(root, "min_max.csv")).values
        self.scenario = pd.read_csv(os.path.join(root, "scenario.csv")).values



    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx) -> torch.tensor:
        input = np.load(self.files[idx])
        input = min_max_scaler(input, self.min_max)
        target = self.target[idx]
        input = torch.tensor(input, dtype=torch.float32)
        if self.classification:
            target = self.scenario[(self.scenario[:, 1] == target[0]) & (self.scenario[:, 2] == target[1]) & (self.scenario[:, 3] == target[2])][:, 0]
            target = torch.tensor(target, dtype=torch.long)
        else:
            target = torch.tensor(target, dtype=torch.float32)
        return input, target


if __name__ == "__main__":
    dataset = DamageIdentificationDataset(path=f"./Data", source=["Displacement", "synthetic"], mode="test", classification=True)
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    for i in dataloader:
        print(i)
