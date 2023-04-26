from dataset.Displacement.SyntheticFinetune import SyntheticDataset
from torch.utils.data import DataLoader
import numpy as np


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
