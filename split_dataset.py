
import os
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

path = "./RawData/Displacement"
train_data  = [] 
train_label = []

valid_data = []
valid_label = []

test_data = []
test_label = []

for situation, signal_name in enumerate(sorted(os.listdir(os.path.join(path, "train")))):
    name, _, _ = scipy.io.whosmat(os.path.join(path, "train", signal_name))[0]
    x = scipy.io.loadmat(os.path.join(path, "train", signal_name))[name]
    label_file = pd.read_csv(os.path.join(path, "label.csv"))
    label = label_file.loc[label_file['File Name'] == signal_name].values[0, 1:4].astype(float)
    length = x.shape[1]
    crop_range = [0, int(0.7 * length), int(0.9 * length), int(length)]
    train_x = sliding_window(x[:, crop_range[0]:crop_range[1]], window=16384, stride=128)
    valid_x = sliding_window(x[:, crop_range[1]:crop_range[2]], window=16384, stride=128)
    test_x = sliding_window(x[:, crop_range[2]:crop_range[3]], window=16384, stride=128)
    



    train_data += train_x
    train_label += [label for i in range(len(train_x))]

    valid_data += valid_x
    valid_label += [label for i in range(len(valid_x))]

    test_data += test_x
    test_label += [label for i in range(len(test_x))]


train_label = np.concatenate([train_label])
valid_label = np.concatenate([valid_label])
test_label = np.concatenate([test_label])

# Prepare training data
if not os.path.exists("./temp/train"):
    os.makedirs("./temp/train")

name = []
for i, data in enumerate(train_data):
    name.append(f"train_{i:05d}.npy")
    np.save(f"./temp/train/train_{i:05d}.npy", data)

df = pd.DataFrame({
    "name": name,
    "label1": train_label[:, 0],
    "label2": train_label[:, 1],
    "label3": train_label[:, 2],
})

df.to_csv("./temp/train.csv", index=False)

# Prepare validation data
if not os.path.exists("./temp/valid"):
    os.makedirs("./temp/valid")

name = []
for i, data in enumerate(valid_data):
    name.append(f"valid_{i:05d}.npy")
    np.save(f"./temp/valid/valid_{i:05d}.npy", data)

df = pd.DataFrame({
    "name": name,
    "label1": valid_label[:, 0],
    "label2": valid_label[:, 1],
    "label3": valid_label[:, 2],
})

df.to_csv("./temp/valid.csv", index=False)


# Prepare testing data
if not os.path.exists("./temp/test"):
    os.makedirs("./temp/test")

name = []
for i, data in enumerate(test_data):
    name.append(f"test_{i:05d}.npy")
    np.save(f"./temp/test/test_{i:05d}.npy", data)

df = pd.DataFrame({
    "name": name,
    "label1": test_label[:, 0],
    "label2": test_label[:, 1],
    "label3": test_label[:, 2],
})

df.to_csv("./temp/test.csv", index=False)
