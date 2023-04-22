import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torch.autograd import grad as torch_grad

import numpy as np
import pandas as pd

from timm.scheduler.cosine_lr import CosineLRScheduler

from pytorch_lightning import LightningModule
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from scipy.special import softmax



class Encoder(nn.Module):
    def __init__(self, length=1024, latent_dim=512):
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(5, 16, 3, 1, 1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, 4, 2, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, 4, 2, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 4, 2, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 4, 2, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 512, 4, 2, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 1024, 32, 1, 0),
        )
    
        

    def forward(self, x):
        x = self.encoder(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(1024, 512, 32, 1, 0),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(512, 256, 4, 2, 1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(256, 128, 4, 2, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(128, 64, 4, 2, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(64, 32, 4, 2, 1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(32, 16, 4, 2, 1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(16, 5, 3, 1, 1),
            nn.Sigmoid()
        )

    
        

    def forward(self, x):
        x = self.decoder(x)
        return x
    
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 18)
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = x.view(x.shape[0], 3, 6)
        return x

class AE(LightningModule):

    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = Encoder()
        self.decoder = Decoder()


    def forward(self, X):
        latent = self.encoder(X)
        reconstruct = self.decoder(latent)
        return latent, reconstruct

    
    def reconstruction_loss(self, 
                           anchor_signal, positive_signal, negative_signal,
                           anchor_reconstruct, positive_reconstruct, negative_reconstruct):
        mse = nn.MSELoss()
        mse_anchor = mse(anchor_signal, anchor_reconstruct)
        mse_positive = mse(positive_signal, positive_reconstruct)
        mse_negative = mse(negative_signal, negative_reconstruct)
        return mse_anchor + mse_positive + mse_negative
    
    def loss_func(self, 
                  anchor_signal, positive_signal, negative_signal,
                  anchor_reconstruct, positive_reconstruct, negative_reconstruct):
        
        mse_loss = self.reconstruction_loss(anchor_signal, positive_signal, negative_signal,
                  anchor_reconstruct, positive_reconstruct, negative_reconstruct)

        return mse_loss

    def training_step(self, batch, batch_idx):
        signal = batch
        representation = self.encoder(signal)
        reconstructed_signal = self.decoder(representation)

        mse = nn.MSELoss()
        mse_loss  = mse(signal, reconstructed_signal)
        loss =  mse_loss

        self.log("train_loss", loss)
        
        self.logger.experiment.add_scalar(f'Learning rate', self.optimizers().param_groups[0]['lr'], self.current_epoch)
        return {"loss": loss, "mse_loss": mse_loss, \
                "anchor_reconstruct": reconstructed_signal, "anchor_signal":signal}

        

    def configure_optimizers(self):
        lr = 5E-5

        opt = torch.optim.Adam(self.parameters(), lr=lr)
        return [opt], []
    
    

    def training_epoch_end(self, training_step_outputs):
        loss = []
        mse_loss = []

        anchor_reconstruct = []
        anchor_signal = []

        for step_result in training_step_outputs:
            loss.append(step_result["loss"].cpu().detach().numpy())
            mse_loss.append(step_result["mse_loss"].cpu().detach().numpy())


            anchor_reconstruct.append(step_result["anchor_reconstruct"].cpu().detach().numpy())
            anchor_signal.append(step_result["anchor_signal"].cpu().detach().numpy())


            
        loss = np.concatenate([loss], axis=0)
        mse_loss = np.concatenate([mse_loss], axis=0)

        self.logger.experiment.add_scalar(f'Train/Loss/Loss', loss.mean(), self.current_epoch)
        self.logger.experiment.add_scalar(f'Train/Loss/MSE Loss', mse_loss.mean(), self.current_epoch)

    
        anchor_reconstruct = np.concatenate(anchor_reconstruct, axis=0)
        anchor_signal = np.concatenate(anchor_signal, axis=0)

        min_max = self.trainer.train_dataloader.dataset.datasets.min_max
        anchor_signal, anchor_reconstruct = self.denormalize(anchor_signal, anchor_reconstruct, min_max)

        self.visualize(anchor_signal[-1], anchor_reconstruct[-1])

    
    


    def visualize(self, signal, reonstructed_signal):
        fig, axes = plt.subplots(5, 1, figsize=(15,6))
        for i in range(5):
            line1 = axes[i].plot(range(len(signal[i, :])), signal[i, :], color="tab:blue",  label="Real Signal")
            line2 = axes[i].plot(range(len(reonstructed_signal[i, :])), reonstructed_signal[i, :], color="tab:red", linestyle='dashed', label="Reconstructed Signal")
            axes[i].set_xticks([])
        fig.legend(handles =[line1[0], line2[0]], loc ='lower center', ncol=4)
        self.logger.experiment.add_figure(f'Train/Visualize', fig , self.current_epoch)

    def denormalize(self, signal, reonstructed_signal, min_max):
        n = signal.shape[0]
        output_signal = np.zeros_like(signal, dtype=float)
        output_reconstruct = np.zeros_like(signal, dtype=float)
        for i in range(n):
            for j in range(5):
                output_reconstruct[i, j, :] = reonstructed_signal[i, j, :] * (min_max[j][1] - min_max[j][0]) + min_max[j][0]
                output_signal[i, j, :] = signal[i, j, :] * (min_max[j][1] - min_max[j][0]) + min_max[j][0]

        return output_signal, output_reconstruct

