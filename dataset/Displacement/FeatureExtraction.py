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
        self.path = os.path.join(path, "Displacement")
        self.triplet = triplet
        self.Damage = Damage
        self.train_path = os.path.join(self.path, "train")
        self.test_path = os.path.join(self.path, "test")
        self.mode = mode

        self.min_max = pd.read_csv(os.path.join(self.path, "min_max.csv")).values
        
        
        if self.mode != "evaluate":
            label_file = pd.read_csv(os.path.join(self.path, "label.csv"))

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
                

                sliding_signal = sliding_window(x, window=1024, stride=1024)
                label = label_file.loc[label_file['File Name'] == signal_name].values[0, 1:4].astype(float)
                label = (label * 10).astype(np.int64)
                label = [label for i in range(len(sliding_signal))]

                train_x, valid_x, train_y, valid_y = train_test_split(sliding_signal, label, train_size=0.7, random_state=0)
                valid_x, test_x, valid_y, test_y = train_test_split(valid_x, valid_y, train_size=(2/3.), random_state=0)
                


                self.train_data += train_x
                self.train_label += train_y
                self.train_id += [situation + 1 for i in range(len(train_x))]

                self.valid_data += valid_x
                self.valid_label += valid_y
                self.valid_id += [situation + 1 for i in range(len(valid_x))]

                self.test_data += test_x
                self.test_label += test_y
                self.test_id += [situation + 1 for i in range(len(test_x))]

            self.train_index = np.array([i for i in range(len(self.train_data))])
            self.train_id = np.array(self.train_id)
            self.valid_id = np.array(self.valid_id)
            self.test_id = np.array(self.test_id)
            

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

            anchor_label = self.train_label[idx]
            anchor_signal = self.train_data[idx]
            anchor_state = self.train_id[idx]

            positive_list = self.train_index[self.train_index!=idx][self.train_id[self.train_index!=idx]==anchor_state]
            positive_item = random.choice(positive_list)
            positive_signal = self.train_data[positive_item]
            positive_label = self.train_label[positive_item]

            negative_list = self.train_index[self.train_index!=idx][self.train_id[self.train_index!=idx]!=anchor_state]
            negative_item = random.choice(negative_list)
            negative_signal = self.train_data[negative_item]
            negative_label = self.train_label[negative_item]

            anchor_signal = torch.tensor(anchor_signal, dtype=torch.float32)
            positive_signal = torch.tensor(positive_signal, dtype=torch.float32)
            negative_signal = torch.tensor(negative_signal, dtype=torch.float32)

            anchor_label = torch.tensor(np.vstack(anchor_label), dtype=torch.long)
            positive_label = torch.tensor(np.vstack(positive_label), dtype=torch.long)
            negative_label = torch.tensor(np.vstack(negative_label), dtype=torch.long)



            return (anchor_signal, positive_signal, negative_signal), \
                (anchor_label, positive_label, negative_label), anchor_state
        
        elif self.mode == "valid":
            anchor_signal = self.valid_data[idx]
            anchor_state = self.valid_id[idx]
            anchor_label = self.valid_label[idx]

            anchor_signal = torch.tensor(anchor_signal, dtype=torch.float32)
            anchor_label = torch.tensor(np.vstack(anchor_label), dtype=torch.long)

            return anchor_signal, anchor_label, anchor_state
        
        elif self.mode == "test":
            anchor_signal = self.test_Data[idx]
            anchor_state = self.test_id[idx]
            anchor_label = self.test_label[idx]

            anchor_signal = torch.tensor(anchor_signal, dtype=torch.float32)
            anchor_label = torch.tensor(np.vstack(anchor_label), dtype=torch.long)

            return anchor_signal, anchor_label, anchor_state

            
            





if __name__ == "__main__":
    dataset = FeatureExtractionDataset(path="./Data", mode="valid")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(next(iter(dataloader))[-2])