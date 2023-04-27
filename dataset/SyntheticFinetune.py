import os
import torch
import scipy
import random
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

from model.Displacement.generation import WCGAN_GP

def min_max_scaler(signal, min_max):
    new_signal = np.copy(signal)
    for i in range(signal.shape[0]):
        new_signal[i, :] = (signal[i, :] - min_max[i][0])/(min_max[i][1] - min_max[i][0])
    return new_signal

def sliding_window(signal, window, stride):
    length = signal.shape[-1]
    i = 0
    x = []
    while i < length-window:
        x.append(signal[:, i:i+window])
        i += stride
    return x


class SyntheticDataset(Dataset):
    def __init__(self, n_times, defined_condition, gan_checkpoint, real_involve=True, mode="train") -> None:
        self.defined_condition = defined_condition
        self.path = "./Data/Displacement"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        gan = WCGAN_GP.load_from_checkpoint(gan_checkpoint).to(self.device)
        self.min_max = pd.read_csv(os.path.join(self.path, "min_max.csv")).values
        label_file = pd.read_csv(os.path.join(self.path, "label.csv"))
        self.mode = mode

        self.train_data = []
        self.train_label = []

        self.valid_data = []
        self.valid_label = []  

        self.test_data = []
        self.test_label = []         

        with torch.no_grad():
            for _ in range(int(0.7*n_times)):
                z = torch.rand((len(defined_condition), 82)).to(self.device)
                input_condition = []
                for condition in defined_condition:
                    self.train_label += [np.array((np.argmax(condition[0])*0.1, np.argmax(condition[1])*0.1, np.argmax(condition[2])*0.1))]
                    input_condition.append(np.concatenate(condition))
                input_condition = torch.tensor(np.array(input_condition), dtype=torch.float32).to(self.device)
                self.train_data.append(gan(z, input_condition).squeeze().cpu().numpy())

            for _ in range(int(0.2*n_times)):
                z = torch.rand((len(defined_condition), 82)).to(self.device)
                input_condition = []
                for condition in defined_condition:
                    self.valid_label += [np.array((np.argmax(condition[0])*0.1, np.argmax(condition[1])*0.1, np.argmax(condition[2])*0.1))]
                    input_condition.append(np.concatenate(condition))
                input_condition = torch.tensor(np.array(input_condition), dtype=torch.float32).to(self.device)
                self.valid_data.append(gan(z, input_condition).squeeze().cpu().numpy())

            for _ in range(int(0.1*n_times)):
                z = torch.rand((len(defined_condition), 82)).to(self.device)
                input_condition = []
                for condition in defined_condition:
                    self.test_label += [np.array((np.argmax(condition[0])*0.1, np.argmax(condition[1])*0.1, np.argmax(condition[2])*0.1))]
                    input_condition.append(np.concatenate(condition))
                input_condition = torch.tensor(np.array(input_condition), dtype=torch.float32).to(self.device)
                self.test_data.append(gan(z, input_condition).squeeze().cpu().numpy())

        if real_involve:
            for situation, signal_name in enumerate(sorted(os.listdir(os.path.join(self.path, "train")))):
                if situation != 10:
                    name, _, _ = scipy.io.whosmat(os.path.join(self.path, "train", signal_name))[0]
                    x = scipy.io.loadmat(os.path.join(self.path, "train", signal_name))[name]
                    x = min_max_scaler(x, self.min_max)
                    label = label_file.loc[label_file['File Name'] == signal_name].values[0, 1:4].astype(float)
                    length = x.shape[1]
                    crop_range = [0, int(0.7 * length), int(0.9 * length), int(length)]
                    train_x = sliding_window(x[:, crop_range[0]:crop_range[1]], window=1024, stride=128)
                    valid_x = sliding_window(x[:, crop_range[1]:crop_range[2]], window=1024, stride=128)
                    test_x = sliding_window(x[:, crop_range[2]:crop_range[3]], window=1024, stride=128)

                    self.train_data.append(np.concatenate([train_x]))
                    self.train_label += [label for _ in range(len(train_x))]
                    self.valid_data.append(np.concatenate([valid_x]))
                    self.valid_label += [label for _ in range(len(valid_x))]
                    self.test_data.append(np.concatenate([test_x]))
                    self.test_label += [label for _ in range(len(test_x))]


        self.train_data = np.concatenate(self.train_data)
        self.valid_data = np.concatenate(self.valid_data)
        self.test_data = np.concatenate(self.test_data)
        self.train_label = np.concatenate([self.train_label])
        self.valid_label = np.concatenate([self.valid_label])
        self.test_label = np.concatenate([self.test_label])
        


    def __len__(self) -> int:
        if self.mode == "train":
            return len(self.train_data)
        elif self.mode == "valid":
            return len(self.valid_data)
        elif self.mode == "test":
            return len(self.test_data)
        else:
            return Exception

    def __getitem__(self, idx) -> torch.tensor:
        if self.mode == "train":
            input = torch.tensor(self.train_data[idx], dtype=torch.float32)
            label = torch.tensor(self.train_label[idx], dtype=torch.float32)
            return input, label

            
        
        elif self.mode == "valid":
            input = torch.tensor(self.valid_data[idx], dtype=torch.float32)
            label = torch.tensor(self.valid_label[idx], dtype=torch.float32)
            return input, label
        
        elif self.mode == "test":
            input = torch.tensor(self.test_data[idx], dtype=torch.float32)
            label = torch.tensor(self.test_label[idx], dtype=torch.float32)
            return input, label

        else:
            input = torch.tensor(self.evaluation_data[idx], dtype=torch.float32)

            return input



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
