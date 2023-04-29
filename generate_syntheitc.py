import os
import torch
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from model.generation import WCGAN_GP

def denormalize(generated_data):
    min_max = pd.read_csv("./Data/Displacement/min_max.csv").values
    n = generated_data.shape[0]
    for i in range(n):
        for j in range(5):
            generated_data[i, j, :] = generated_data[i, j, :] * (min_max[j, 1] - min_max[j, 0]) + min_max[j, 0]

    return generated_data

WCGAN = WCGAN_GP.load_from_checkpoint("./Logs/Generation/Displacement_mean_constraint/LAST/version_0/checkpoints/epoch=00199.ckpt", strict=False).to("cuda")

all_condition = []
for d7 in range(6):
    for d22 in range(6):
        for d38 in range(6):
            onehot7 = np.eye(6)[d7]
            onehot22 = np.eye(6)[d22]
            onehot38 = np.eye(6)[d38]
            all_condition.append((onehot7, onehot22, onehot38))

n_times = 100
label = []
data = []

with torch.no_grad():
    for _ in range(int(n_times)):
        z = torch.rand((len(all_condition), 82)).to("cuda")
        input_condition = []
        temp_label = []
        for condition in all_condition:
            temp_label += [np.array((round(np.argmax(condition[0])*0.1, 1), round(np.argmax(condition[1])*0.1, 1), round(np.argmax(condition[2])*0.1, 1)))]
            input_condition.append(np.concatenate(condition))
        label.append(temp_label)
        input_condition = torch.tensor(np.array(input_condition), dtype=torch.float32).to("cuda")
        data.append(denormalize(WCGAN(z, input_condition).squeeze().cpu().numpy()))

train_x, temp_x, train_y, temp_y = train_test_split(data, label, train_size=0.7, random_state=0)
valid_x, test_x, valid_y, test_y = train_test_split(temp_x, temp_y, train_size=0.7, random_state=0)


## Saving training data
train_data = np.concatenate(train_x)
train_label = np.concatenate(train_y)

if not os.path.exists("./Data/synthetic/train"):
    os.makedirs("./Data/synthetic/train")

name_list = []
for i in range(train_data.shape[0]):
    name = f"train_{i:05d}.npy"
    name_list.append(name)
    np.save(f"./Data/synthetic/train/{name}", train_data[i, :, :])
df = pd.DataFrame({
    "name":name_list,
    "label1": train_label[:, 0],
    "label2": train_label[:, 1],
    "label3": train_label[:, 2],
})

df.to_csv(f"./Data/synthetic/train.csv", index=False)

## Saving validation data
valid_data = np.concatenate(valid_x)
valid_label = np.concatenate(valid_y)

if not os.path.exists("./Data/synthetic/valid"):
    os.makedirs("./Data/synthetic/valid")

name_list = []
for i in range(valid_data.shape[0]):
    name = f"valid_{i:05d}.npy"
    name_list.append(name)
    np.save(f"./Data/synthetic/valid/{name}", valid_data[i, :, :])
df = pd.DataFrame({
    "name":name_list,
    "label1": valid_label[:, 0],
    "label2": valid_label[:, 1],
    "label3": valid_label[:, 2],
})

df.to_csv(f"./Data/synthetic/valid.csv", index=False)

## Saving testing data
test_data = np.concatenate(test_x)
test_label = np.concatenate(test_y)

if not os.path.exists("./Data/synthetic/test"):
    os.makedirs("./Data/synthetic/test")

name_list = []
for i in range(test_data.shape[0]):
    name = f"test_{i:05d}.npy"
    name_list.append(name)
    np.save(f"./Data/synthetic/test/{name}", test_data[i, :, :])
df = pd.DataFrame({
    "name":name_list,
    "label1": test_label[:, 0],
    "label2": test_label[:, 1],
    "label3": test_label[:, 2],
})

df.to_csv(f"./Data/synthetic/test.csv", index=False)