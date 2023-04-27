from model.generation import WCGAN_GP
import numpy as np
import torch

WCGAN = WCGAN_GP.load_from_checkpoint("./Logs/Generation/Displacement_mean_constraint/LAST/version_0/checkpoints/epoch=00199.ckpt", strict=False).to("cuda")

all_condition = []
for d7 in range(6):
    for d22 in range(6):
        for d38 in range(6):
            onehot7 = np.eye(6)[d7]
            onehot22 = np.eye(6)[d22]
            onehot38 = np.eye(6)[d38]
            all_condition.append((onehot7, onehot22, onehot38))

n_times = 50
train_label = []
train_data = []

with torch.no_grad():
    for _ in range(int(n_times)):
        z = torch.rand((len(all_condition), 82)).to("cuda")
        input_condition = []
        for condition in all_condition:
            train_label += [np.array((np.argmax(condition[0])*0.1, np.argmax(condition[1])*0.1, np.argmax(condition[2])*0.1))]
            input_condition.append(np.concatenate(condition))
        input_condition = torch.tensor(np.array(input_condition), dtype=torch.float32).to("cuda")
        train_data.append(WCGAN(z, input_condition).squeeze().cpu().numpy())

train_data = np.concatenate(train_data)
train_label = np.concatenate([train_label])

for i in range(train_data.shape[0]):
    np.save(train_data)

print(i)