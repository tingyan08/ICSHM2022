import os
import torch
import scipy
import random
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

from model.Displacement.generation import WCGAN_GP



class SyntheticDataset(Dataset):
    def __init__(self, n_times, defined_condition, gan_checkpoint, real_involve=False) -> None:
        self.defined_condition = defined_condition

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gan = WCGAN_GP.load_from_checkpoint(gan_checkpoint).to(self.device)

        self.data = []
        self.conditions = []
        self.labels = []
        with torch.no_grad():
            for _ in range(n_times):
                z = torch.rand((len(defined_condition), 82)).to(self.device)
                input_condition = []
                for condition in defined_condition:
                    self.labels.append(condition)
                    self.conditions.append(np.concatenate(condition))
                    input_condition.append(np.concatenate(condition))
                input_condition = torch.tensor(np.array(input_condition), dtype=torch.float32).to(self.device)
                self.data.append(self.gan(z, input_condition).squeeze().cpu().numpy())

        self.data = np.concatenate(self.data)
        self.conditions = np.concatenate([self.conditions])
        


    def __len__(self) -> int:
        return self.conditions.shape[0]
        # return 512

    def __getitem__(self, index) -> torch.tensor:
        input = self.data[index]
        condition = self.conditions[index]
        label = self.labels[index]

        condition = torch.tensor(condition, dtype=torch.float32)
        return input, condition, label[0], label[1], label[2]


if __name__ == "__main__":
    all_condition = []
    for d7 in range(6):
        for d22 in range(6):
            for d38 in range(6):
                onehot7 = np.eye(6)[d7]
                onehot22 = np.eye(6)[d22]
                onehot38 = np.eye(6)[d38]
                all_condition.append((onehot7, onehot22, onehot38))

    dataset = SyntheticDataset(n_times=50, defined_condition=all_condition, gan_checkpoint="./Logs/Generation/Displacement_WCGAN_GP/stride_dataset/version_0/checkpoints/epoch=00499.ckpt")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    a = next(dataloader.__iter__())
    print(a)
