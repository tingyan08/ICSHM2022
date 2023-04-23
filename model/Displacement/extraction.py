import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import numpy as np
import pandas as pd

from timm.scheduler.cosine_lr import CosineLRScheduler

from pytorch_lightning import LightningModule
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from scipy.special import softmax

from ..utils import DoubleConv, Down, Up, OutConv



class Encoder(nn.Module):
    def __init__(self, bilinear = False):
        super(Encoder, self).__init__()
        self.input_conv = nn.Sequential(
            nn.Conv1d(5, 16, 3, 1, 1),
        )
        self.inc = (DoubleConv(16, 16))
        self.down1 = (Down(16, 32))
        self.down2 = (Down(32, 64))
        self.down3 = (Down(64, 128))
        self.down4 = (Down(128, 256) )
        self.down5 = (Down(256, 512) )
        self.down6 = (Down(512, 1024) )
        self.linear = nn.Linear(16, 1)


    def forward(self, x):
        x = self.input_conv(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        x7 = self.linear(x7)
        return x1, x2, x3, x4, x5, x6, x7
    
class Decoder(nn.Module):
    def __init__(self, bilinear = False):
        super(Decoder, self).__init__()
        self.linear = nn.Linear(1, 16)
        factor = 2 if bilinear else 1
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.up5 = (Up(64, 32, bilinear))
        self.up6 = (Up(32, 16, bilinear))
        self.outc = (OutConv(16, 16))
        self.output_conv = nn.Sequential(
            nn.Conv1d(16, 5, 3, 1, 1), 
        )

    def forward(self, latents):
        x1, x2, x3, x4, x5, x6, x7 = latents
        x7 = self.linear(x7)
        x = self.up1(x7, x6)
        x = self.up2(x, x5)
        x = self.up3(x, x4)
        x = self.up4(x, x3)
        x = self.up5(x, x2)
        x = self.up6(x, x1)
        x = self.outc(x)
        logits = self.output_conv(x)
        return logits
    
    
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
        return x[:, :6], x[:, 6:12], x[:, 12:18]
    

class AE(LightningModule):

    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = Encoder()
        self.decoder = Decoder()


    def forward(self, X):
        latents = self.encoder(X)
        reconstruct = self.decoder(latents)
        return reconstruct, latents[-1]

    
    def reconstruction_loss(self, 
                           anchor_signal, positive_signal, negative_signal,
                           anchor_reconstruct, positive_reconstruct, negative_reconstruct):
        mse = nn.MSELoss()
        mse_anchor = mse(anchor_signal, anchor_reconstruct)
        mse_positive = mse(positive_signal, positive_reconstruct)
        mse_negative = mse(negative_signal, negative_reconstruct)
        return mse_anchor + mse_positive + mse_negative
    
    def loss_func(self, 
                  anchor_signal, 
                  anchor_reconstruct):
        mse = nn.MSELoss()
        mse_loss = mse(anchor_signal, anchor_reconstruct)

        return mse_loss

    def training_step(self, batch, batch_idx):
        signal_list, _, anchor_state = batch
        anchor_signal, _, _ = signal_list[0], signal_list[1], signal_list[2]


        latents = self.encoder(anchor_signal)
        anchor_representation = latents[-1]
        anchor_reconstruct = self.decoder(latents)

        mse_loss  = self.loss_func(anchor_signal,anchor_reconstruct)

        loss =  mse_loss

        
        self.logger.experiment.add_scalar(f'Learning rate', self.optimizers().param_groups[0]['lr'], self.current_epoch)
        return {"loss": loss, "mse_loss": mse_loss, \
                "anchor_representation":torch.permute(anchor_representation, (0, 2, 1)), "anchor_state":anchor_state, \
                "anchor_reconstruct": anchor_reconstruct, "anchor_signal":anchor_signal}


    def validation_step(self, batch, batch_idx):
        anchor_signal, _, anchor_state = batch

        latents = self.encoder(anchor_signal)
        anchor_representation = latents[-1]
        anchor_reconstruct = self.decoder(latents)

        mse_loss  = self.loss_func(anchor_signal, anchor_reconstruct)

        loss =  mse_loss

        
        return {"loss": loss, "mse_loss": mse_loss, \
                "anchor_representation":torch.permute(anchor_representation, (0, 2, 1)), "anchor_state":anchor_state, \
                "anchor_reconstruct": anchor_reconstruct, "anchor_signal":anchor_signal}
        

    def configure_optimizers(self):
        lr = 5E-5

        opt = torch.optim.Adam(self.parameters(), lr=lr)
        return [opt], []
    
    def training_epoch_end(self, training_step_outputs):
        loss = []
        mse_loss = []

        anchor_representation = []
        anchor_state = []

        anchor_reconstruct = []
        anchor_signal = []

        for step_result in training_step_outputs:
            loss.append(step_result["loss"].cpu().detach().numpy())
            mse_loss.append(step_result["mse_loss"].cpu().detach().numpy())

            anchor_representation.append(step_result["anchor_representation"].cpu().detach().numpy())
            anchor_state.append(step_result["anchor_state"].cpu().detach().numpy())

            anchor_reconstruct.append(step_result["anchor_reconstruct"].cpu().detach().numpy())
            anchor_signal.append(step_result["anchor_signal"].cpu().detach().numpy())

        loss = np.concatenate([loss], axis=0)
        mse_loss = np.concatenate([mse_loss], axis=0)
        
        self.logger.experiment.add_scalar(f'Train/Loss/Loss', loss.mean(), self.current_epoch)
        self.logger.experiment.add_scalar(f'Train/Loss/MSE Loss', mse_loss.mean(), self.current_epoch)

        anchor_representation = np.concatenate(anchor_representation, axis=0)
        anchor_state = np.concatenate(anchor_state, axis=0)

        self.draw_tsne(anchor_representation, anchor_state, "Train")
        self.draw_pca(anchor_representation, anchor_state, "Train")

        anchor_reconstruct = np.concatenate(anchor_reconstruct, axis=0)
        anchor_signal = np.concatenate(anchor_signal, axis=0)

        min_max = self.trainer.val_dataloaders[0].dataset.min_max
        anchor_signal, anchor_reconstruct = self.denormalize(anchor_signal, anchor_reconstruct, min_max)

        self.visualize(anchor_signal[-1], anchor_reconstruct[-1], "Train")

    def validation_epoch_end(self, validation_step_outputs):
        loss = []
        mse_loss = []

        anchor_representation = []
        anchor_state = []

        anchor_reconstruct = []
        anchor_signal = []

        for step_result in validation_step_outputs:
            loss.append(step_result["loss"].cpu().numpy())
            mse_loss.append(step_result["mse_loss"].cpu().numpy())

            anchor_representation.append(step_result["anchor_representation"].cpu().numpy())
            anchor_state.append(step_result["anchor_state"].cpu().numpy())

            anchor_reconstruct.append(step_result["anchor_reconstruct"].cpu().numpy())
            anchor_signal.append(step_result["anchor_signal"].cpu().numpy())


            
        loss = np.concatenate([loss], axis=0)
        mse_loss = np.concatenate([mse_loss], axis=0)

        self.logger.experiment.add_scalar(f'Validation/Loss/Loss', loss.mean(), self.current_epoch)
        self.logger.experiment.add_scalar(f'Validation/Loss/MSE Loss', mse_loss.mean(), self.current_epoch)

        self.log('val_loss', mse_loss.mean())

        anchor_representation = np.concatenate(anchor_representation, axis=0)
        anchor_state = np.concatenate(anchor_state, axis=0)

        self.draw_tsne(anchor_representation, anchor_state, "Validation")
        self.draw_pca(anchor_representation, anchor_state, "Validation")

        anchor_reconstruct = np.concatenate(anchor_reconstruct, axis=0)
        anchor_signal = np.concatenate(anchor_signal, axis=0)

        min_max = self.trainer.val_dataloaders[0].dataset.min_max
        anchor_signal, anchor_reconstruct = self.denormalize(anchor_signal, anchor_reconstruct, min_max)

        self.visualize(anchor_signal[-1], anchor_reconstruct[-1], "Validation")

    
    
    def draw_pca(self, anchor, anchor_label, mode):
        anchor = anchor.squeeze()
        embedded = PCA(n_components=2).fit_transform(anchor)
        fig = plt.figure(figsize=(15,10))
        plt.scatter(embedded[:, 0], embedded[:, 1], c=anchor_label, marker=".", alpha=0.6, cmap="Set3")
        self.logger.experiment.add_figure(f'{mode}/PCA', fig , self.current_epoch)
    
    def draw_tsne(self, anchor, anchor_label, mode):
        anchor = anchor.squeeze()
        embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(anchor)
        fig = plt.figure(figsize=(15,10))
        plt.scatter(embedded[:, 0], embedded[:, 1], c=anchor_label, marker=".", alpha=0.6, cmap="Set3")
        self.logger.experiment.add_figure(f'{mode}/TSNE', fig , self.current_epoch)

    def visualize(self, signal, reonstructed_signal, mode):
        fig, axes = plt.subplots(5, 1, figsize=(15,6))
        for i in range(5):
            line1 = axes[i].plot(range(len(signal[i, :])), signal[i, :], color="tab:blue",  label="Real Signal")
            line2 = axes[i].plot(range(len(reonstructed_signal[i, :])), reonstructed_signal[i, :], color="tab:red", linestyle='dashed', label="Reconstructed Signal")
            axes[i].set_xticks([])
        fig.legend(handles =[line1[0], line2[0]], loc ='lower center', ncol=4)
        self.logger.experiment.add_figure(f'{mode}/Visualize', fig , self.current_epoch)

    def denormalize(self, signal, reonstructed_signal, min_max):
        n = signal.shape[0]
        output_signal = np.zeros_like(signal, dtype=float)
        output_reconstruct = np.zeros_like(signal, dtype=float)
        for i in range(n):
            for j in range(5):
                output_reconstruct[i, j, :] = reonstructed_signal[i, j, :] * (min_max[j][1] - min_max[j][0]) + min_max[j][0]
                output_signal[i, j, :] = signal[i, j, :] * (min_max[j][1] - min_max[j][0]) + min_max[j][0]

        return output_signal, output_reconstruct

class TripletAE(LightningModule):

    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = Encoder()
        self.decoder = Decoder()



    def forward(self, X):
        latents = self.encoder(X)
        reconstruct = self.decoder(latents)
        return reconstruct, latents[-1]


    
    def triplet_loss(self, anchor, positive, negative):
        anchor = torch.permute(anchor, (0, 2, 1))
        positive = torch.permute(positive, (0, 2, 1))
        negative = torch.permute(negative, (0, 2, 1))
        loss_func = nn.TripletMarginLoss(margin=1.0, p=2)
        triplet_loss = loss_func(anchor, positive, negative)
        return triplet_loss
    
    def reconstruction_loss(self, 
                           anchor_signal, positive_signal, negative_signal,
                           anchor_reconstruct, positive_reconstruct, negative_reconstruct):
        mse = nn.MSELoss()
        mse_anchor = mse(anchor_signal, anchor_reconstruct)
        mse_positive = mse(positive_signal, positive_reconstruct)
        mse_negative = mse(negative_signal, negative_reconstruct)
        return mse_anchor + mse_positive + mse_negative
    
    def loss_func(self, 
                  anchor_representation, positive_representation, negative_representation,
                  anchor_signal, positive_signal, negative_signal,
                  anchor_reconstruct, positive_reconstruct, negative_reconstruct):
        
        mse_loss = self.reconstruction_loss(anchor_signal, positive_signal, negative_signal,
                  anchor_reconstruct, positive_reconstruct, negative_reconstruct)
        triplet_loss = self.triplet_loss(anchor_representation, positive_representation, negative_representation)

        return triplet_loss, mse_loss

    def training_step(self, batch, batch_idx):
        signal_list, _, anchor_state = batch
        anchor_signal, positive_signal, negative_signal = signal_list[0], signal_list[1], signal_list[2]


        anchor_latents = self.encoder(anchor_signal)
        positive_latents = self.encoder(positive_signal)
        negative_latents = self.encoder(negative_signal)

        anchor_reconstruct = self.decoder(anchor_latents)
        positive_reconstruct = self.decoder(positive_latents)
        negative_reconstruct = self.decoder(negative_latents)

        anchor_representation = anchor_latents[-1]
        positive_representation = positive_latents[-1]
        negative_representation = negative_latents[-1]

        triplet_loss, mse_loss  = self.loss_func(
            anchor_representation, positive_representation, negative_representation,
            anchor_signal, positive_signal, negative_signal,
            anchor_reconstruct, positive_reconstruct, negative_reconstruct
        )

        loss = triplet_loss + 1.2 * mse_loss

        self.log("train_loss", loss)
        
        self.logger.experiment.add_scalar(f'Learning rate', self.optimizers().param_groups[0]['lr'], self.current_epoch)
        return {"loss": loss, "triplet_loss": triplet_loss, "mse_loss": mse_loss, \
                "anchor_representation":torch.permute(anchor_representation, (0, 2, 1)), "anchor_state":anchor_state, \
                "anchor_reconstruct": anchor_reconstruct, "anchor_signal":anchor_signal}
    
    def validation_step(self, batch, batch_idx):
        anchor_signal, _, anchor_state = batch


        latents = self.encoder(anchor_signal)
        anchor_reconstruct = self.decoder(latents)
        anchor_representation = latents[-1]

        mse = nn.MSELoss()
        mse_loss  = mse(anchor_signal, anchor_reconstruct)

        loss = mse_loss

        self.log("train_loss", loss)
        
        return {"loss": loss, "mse_loss": mse_loss, \
                "anchor_representation":torch.permute(anchor_representation, (0, 2, 1)), "anchor_state":anchor_state, \
                "anchor_reconstruct": anchor_reconstruct, "anchor_signal":anchor_signal}

    def configure_optimizers(self):
        lr = 5E-5

        opt = torch.optim.Adam(self.parameters(), lr=lr)
        return [opt], []
    
    def training_epoch_end(self, training_step_outputs):
        loss = []
        triplet_loss = []
        mse_loss = []

        anchor_representation = []
        anchor_state = []

        anchor_reconstruct = []
        anchor_signal = []

        for step_result in training_step_outputs:
            loss.append(step_result["loss"].cpu().detach().numpy())
            triplet_loss.append(step_result["triplet_loss"].cpu().detach().numpy())
            mse_loss.append(step_result["mse_loss"].cpu().detach().numpy())
            
            anchor_representation.append(step_result["anchor_representation"].cpu().detach().numpy())
            anchor_state.append(step_result["anchor_state"].cpu().detach().numpy())

            anchor_reconstruct.append(step_result["anchor_reconstruct"].cpu().detach().numpy())
            anchor_signal.append(step_result["anchor_signal"].cpu().detach().numpy())
            

        loss = np.concatenate([loss], axis=0)
        triplet_loss = np.concatenate([triplet_loss], axis=0)
        mse_loss = np.concatenate([mse_loss], axis=0)

        self.logger.experiment.add_scalar(f'Train/Loss/Loss', loss.mean(), self.current_epoch)
        self.logger.experiment.add_scalar(f'Train/Loss/Triplet Loss', triplet_loss.mean(), self.current_epoch)
        self.logger.experiment.add_scalar(f'Train/Loss/MSE Loss', mse_loss.mean(), self.current_epoch)

        anchor_representation = np.concatenate(anchor_representation, axis=0)
        anchor_state = np.concatenate(anchor_state, axis=0)

        self.draw_tsne(anchor_representation, anchor_state, "Train")
        self.draw_pca(anchor_representation, anchor_state, "Train")

        anchor_reconstruct = np.concatenate(anchor_reconstruct, axis=0)
        anchor_signal = np.concatenate(anchor_signal, axis=0)

        min_max = self.trainer.val_dataloaders[0].dataset.min_max
        anchor_signal, anchor_reconstruct = self.denormalize(anchor_signal, anchor_reconstruct, min_max)

        self.visualize(anchor_signal[-1], anchor_reconstruct[-1], "Train")



    def validation_epoch_end(self, validation_step_outputs):
        loss = []
        mse_loss = []

        anchor_representation = []
        anchor_state = []

        anchor_reconstruct = []
        anchor_signal = []

        for step_result in validation_step_outputs:
            loss.append(step_result["loss"].cpu().numpy())
            mse_loss.append(step_result["mse_loss"].cpu().numpy())

            anchor_representation.append(step_result["anchor_representation"].cpu().numpy())
            anchor_state.append(step_result["anchor_state"].cpu().numpy())

            anchor_reconstruct.append(step_result["anchor_reconstruct"].cpu().numpy())
            anchor_signal.append(step_result["anchor_signal"].cpu().numpy())
            
        loss = np.concatenate([loss], axis=0)
        mse_loss = np.concatenate([mse_loss], axis=0)

        self.logger.experiment.add_scalar(f'Validation/Loss/Loss', loss.mean(), self.current_epoch)
        self.logger.experiment.add_scalar(f'Validation/Loss/MSE Loss', mse_loss.mean(), self.current_epoch)

        self.log("val_loss", mse_loss.mean())

        anchor_representation = np.concatenate(anchor_representation, axis=0)
        anchor_state = np.concatenate(anchor_state, axis=0)

        self.draw_tsne(anchor_representation, anchor_state, "Validation")
        self.draw_pca(anchor_representation, anchor_state, "Validation")


        anchor_reconstruct = np.concatenate(anchor_reconstruct, axis=0)
        anchor_signal = np.concatenate(anchor_signal, axis=0)

        min_max = self.trainer.val_dataloaders[0].dataset.min_max
        anchor_signal, anchor_reconstruct = self.denormalize(anchor_signal, anchor_reconstruct, min_max)

        self.visualize(anchor_signal[-1], anchor_reconstruct[-1], "Validation")

    
    
    def draw_pca(self, anchor, anchor_label, mode):
        anchor = anchor.squeeze()
        embedded = PCA(n_components=2).fit_transform(anchor)
        fig = plt.figure(figsize=(15,10))
        plt.scatter(embedded[:, 0], embedded[:, 1], c=anchor_label, marker=".", alpha=0.6, cmap="Set3")
        self.logger.experiment.add_figure(f'{mode}/PCA', fig , self.current_epoch)
    
    def draw_tsne(self, anchor, anchor_label, mode):
        anchor = anchor.squeeze()
        embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(anchor)
        fig = plt.figure(figsize=(15,10))
        plt.scatter(embedded[:, 0], embedded[:, 1], c=anchor_label, marker=".", alpha=0.6, cmap="Set3")
        self.logger.experiment.add_figure(f'{mode}/TSNE', fig , self.current_epoch)

    def visualize(self, signal, reonstructed_signal, mode):
        fig, axes = plt.subplots(5, 1, figsize=(15,6))
        for i in range(5):
            line1 = axes[i].plot(range(len(signal[i, :])), signal[i, :], color="tab:blue",  label="Real Signal")
            line2 = axes[i].plot(range(len(reonstructed_signal[i, :])), reonstructed_signal[i, :], color="tab:red", linestyle='dashed', label="Reconstructed Signal")
            axes[i].set_xticks([])
        fig.legend(handles =[line1[0], line2[0]], loc ='lower center', ncol=4)
        self.logger.experiment.add_figure(f'{mode}/Visualize', fig , self.current_epoch)

    def denormalize(self, signal, reonstructed_signal, min_max):
        n = signal.shape[0]
        output_signal = np.zeros_like(signal, dtype=float)
        output_reconstruct = np.zeros_like(signal, dtype=float)
        for i in range(n):
            for j in range(5):
                output_reconstruct[i, j, :] = reonstructed_signal[i, j, :] * (min_max[j][1] - min_max[j][0]) + min_max[j][0]
                output_signal[i, j, :] = signal[i, j, :] * (min_max[j][1] - min_max[j][0]) + min_max[j][0]

        return output_signal, output_reconstruct

class DamageAE(LightningModule):

    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = Encoder()
        self.decoder = Decoder()
        self.classifier = Classifier()



    def forward(self, X):
        latents = self.encoder(X)
        reconstruct = self.decoder(latents)
        pred = self.classifier(latents[-1])
        return reconstruct, latents[-1], pred

    def reconstruction_loss(self, 
                           anchor_signal, positive_signal, negative_signal,
                           anchor_reconstruct, positive_reconstruct, negative_reconstruct):
        mse = nn.MSELoss()
        mse_anchor = mse(anchor_signal, anchor_reconstruct)
        mse_positive = mse(positive_signal, positive_reconstruct)
        mse_negative = mse(negative_signal, negative_reconstruct)
        return mse_anchor + mse_positive + mse_negative
    
    def sensor_ce_loss(self, damage_pred, damage_target):
        loss = torch.zeros(1).to(self.device)
        ce = nn.CrossEntropyLoss()
        for i in range(3):
            pred = damage_pred[:, i, :]
            target = damage_target[:, i, 0]
            loss += ce(pred, target)

        return loss

    def classification_loss(self, pred, target):
        anchor_loss = self.sensor_ce_loss(pred[0], target[0])
        positive_loss = self.sensor_ce_loss(pred[1], target[1])
        negative_loss = self.sensor_ce_loss(pred[2], target[2])
        return anchor_loss + positive_loss + negative_loss
    

    def training_step(self, batch, batch_idx):
        signal_list, target_list, anchor_state = batch
        anchor_signal, positive_signal, negative_signal = signal_list[0], signal_list[1], signal_list[2]
        anchor_target, positive_target, negative_target = target_list[0], target_list[1], target_list[2]


        anchor_latents = self.encoder(anchor_signal)
        positive_latents = self.encoder(positive_signal)
        negative_latents = self.encoder(negative_signal)

        anchor_reconstruct = self.decoder(anchor_latents)
        positive_reconstruct = self.decoder(positive_latents)
        negative_reconstruct = self.decoder(negative_latents)

        anchor_representation = anchor_latents[-1]
        positive_representation = positive_latents[-1]
        negative_representation = negative_latents[-1]

        anchor_pred = self.classifier(anchor_representation)
        anchor_pred = torch.concatenate([i.unsqueeze(1) for i in anchor_pred], axis=1)
        positive_pred = self.classifier(positive_representation)
        positive_pred = torch.concatenate([i.unsqueeze(1) for i in positive_pred], axis=1)
        negative_pred = self.classifier(negative_representation)
        negative_pred = torch.concatenate([i.unsqueeze(1) for i in negative_pred], axis=1)


        mse_loss  = self.reconstruction_loss(
            anchor_signal, positive_signal, negative_signal,
            anchor_reconstruct, positive_reconstruct, negative_reconstruct
        )

        ce_loss = self.classification_loss(
            [anchor_pred, positive_pred, negative_pred], [anchor_target, positive_target, negative_target])


        loss = mse_loss + 0.5 * ce_loss

        self.logger.experiment.add_scalar(f'Learning rate', self.optimizers().param_groups[0]['lr'], self.current_epoch)
        return {"loss": loss, "mse_loss": mse_loss, "ce_loss":ce_loss, \
                "anchor_representation":torch.permute(anchor_representation, (0, 2, 1)), "anchor_state":anchor_state, \
                "anchor_reconstruct": anchor_reconstruct, "anchor_signal":anchor_signal,\
                    "anchor_label": anchor_target, "anchor_pred":anchor_pred}
    
    def validation_step(self, batch, batch_idx):
        anchor_signal, anchor_label, anchor_state = batch
        latents = self.encoder(anchor_signal)
        anchor_reconstruct = self.decoder(latents)
        anchor_representation = latents[-1]
        anchor_pred = self.classifier(anchor_representation)
        anchor_pred = torch.concatenate([i.unsqueeze(1) for i in anchor_pred], axis=1)

        mse = nn.MSELoss()
        mse_loss  = mse(anchor_signal, anchor_reconstruct)

        ce_loss = self.sensor_ce_loss(anchor_pred, anchor_label)

        loss = mse_loss + ce_loss

        self.log("val_loss", loss)
        self.logger.experiment.add_scalar(f'Learning rate', self.optimizers().param_groups[0]['lr'], self.current_epoch)
        return {"loss": loss, "mse_loss": mse_loss, "ce_loss":ce_loss, \
                "anchor_representation":torch.permute(anchor_representation, (0, 2, 1)), "anchor_state":anchor_state, \
                "anchor_reconstruct": anchor_reconstruct, "anchor_signal":anchor_signal,\
                "anchor_label": anchor_label, "anchor_pred":anchor_pred}

        

    def configure_optimizers(self):
        lr = 5E-5

        opt = torch.optim.Adam(self.parameters(), lr=lr)
        return [opt], []
    
    def training_epoch_end(self, training_step_outputs):
        loss = []
        ce_loss = []
        mse_loss = []

        anchor_representation = []
        anchor_state = []

        anchor_reconstruct = []
        anchor_signal = []

        anchor_label = []
        anchor_pred = []

        for step_result in training_step_outputs:
            loss.append(step_result["loss"].cpu().detach().numpy())
            ce_loss.append(step_result["ce_loss"].cpu().detach().numpy())
            mse_loss.append(step_result["mse_loss"].cpu().detach().numpy())

            anchor_representation.append(step_result["anchor_representation"].cpu().detach().numpy())
            anchor_state.append(step_result["anchor_state"].cpu().detach().numpy())

            anchor_reconstruct.append(step_result["anchor_reconstruct"].cpu().detach().numpy())
            anchor_signal.append(step_result["anchor_signal"].cpu().detach().numpy())

            anchor_label.append(step_result["anchor_label"].cpu().detach().numpy())
            anchor_pred.append(step_result["anchor_pred"].cpu().detach().numpy())
      
        loss = np.concatenate([loss], axis=0)
        ce_loss = np.concatenate([ce_loss], axis=0)
        mse_loss = np.concatenate([mse_loss], axis=0)

        self.logger.experiment.add_scalar(f'Train/Loss/Loss', loss.mean(), self.current_epoch)
        self.logger.experiment.add_scalar(f'Train/Loss/CE Loss', ce_loss.mean(), self.current_epoch)
        self.logger.experiment.add_scalar(f'Train/Loss/MSE Loss', mse_loss.mean(), self.current_epoch)

        anchor_representation = np.concatenate(anchor_representation, axis=0)
        anchor_state = np.concatenate(anchor_state, axis=0)

        self.draw_tsne(anchor_representation, anchor_state, "Train")
        self.draw_pca(anchor_representation, anchor_state, "Train")

        

        anchor_reconstruct = np.concatenate(anchor_reconstruct, axis=0)
        anchor_signal = np.concatenate(anchor_signal, axis=0)

        min_max = self.trainer.val_dataloaders[0].dataset.min_max
        anchor_signal, anchor_reconstruct = self.denormalize(anchor_signal, anchor_reconstruct, min_max)

        self.visualize(anchor_signal[-1], anchor_reconstruct[-1], "Train")

        anchor_label = np.concatenate(anchor_label, axis=0)
        anchor_pred = np.concatenate(anchor_pred, axis=0)
        self.display_prediction(anchor_label, anchor_pred)


    def validation_epoch_end(self, validation_step_outputs):
        loss = []
        ce_loss = []
        mse_loss = []

        anchor_representation = []
        anchor_state = []

        anchor_reconstruct = []
        anchor_signal = []

        anchor_label = []
        anchor_pred = []

        for step_result in validation_step_outputs:
            loss.append(step_result["loss"].cpu().numpy())
            ce_loss.append(step_result["ce_loss"].cpu().numpy())
            mse_loss.append(step_result["mse_loss"].cpu().numpy())

            anchor_representation.append(step_result["anchor_representation"].cpu().numpy())
            anchor_state.append(step_result["anchor_state"].cpu().numpy())

            anchor_reconstruct.append(step_result["anchor_reconstruct"].cpu().numpy())
            anchor_signal.append(step_result["anchor_signal"].cpu().numpy())

            anchor_label.append(step_result["anchor_label"].cpu().numpy())
            anchor_pred.append(step_result["anchor_pred"].cpu().numpy())
      
        loss = np.concatenate([loss], axis=0)
        ce_loss = np.concatenate([ce_loss], axis=0)
        mse_loss = np.concatenate([mse_loss], axis=0)

        self.logger.experiment.add_scalar(f'Validation/Loss/Loss', loss.mean(), self.current_epoch)
        self.logger.experiment.add_scalar(f'Validation/Loss/CE Loss', ce_loss.mean(), self.current_epoch)
        self.logger.experiment.add_scalar(f'Validation/Loss/MSE Loss', mse_loss.mean(), self.current_epoch)

        anchor_representation = np.concatenate(anchor_representation, axis=0)
        anchor_state = np.concatenate(anchor_state, axis=0)

        self.draw_tsne(anchor_representation, anchor_state, "Validation")
        self.draw_pca(anchor_representation, anchor_state, "Validation")

        

        anchor_reconstruct = np.concatenate(anchor_reconstruct, axis=0)
        anchor_signal = np.concatenate(anchor_signal, axis=0)

        min_max = self.trainer.val_dataloaders[0].dataset.min_max
        anchor_signal, anchor_reconstruct = self.denormalize(anchor_signal, anchor_reconstruct, min_max)

        self.visualize(anchor_signal[-1], anchor_reconstruct[-1], "Validation")

        anchor_label = np.concatenate(anchor_label, axis=0)
        anchor_pred = np.concatenate(anchor_pred, axis=0)
        self.display_prediction(anchor_label, anchor_pred)

    
    
    def draw_pca(self, anchor, anchor_label, mode):
        anchor = anchor.squeeze()
        embedded = PCA(n_components=2).fit_transform(anchor)
        fig = plt.figure(figsize=(15,10))
        plt.scatter(embedded[:, 0], embedded[:, 1], c=anchor_label, marker=".", alpha=0.6, cmap="Set3")
        self.logger.experiment.add_figure(f'{mode}/PCA', fig , self.current_epoch)
    
    def draw_tsne(self, anchor, anchor_label, mode):
        anchor = anchor.squeeze()
        embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(anchor)
        fig = plt.figure(figsize=(15,10))
        plt.scatter(embedded[:, 0], embedded[:, 1], c=anchor_label, marker=".", alpha=0.6, cmap="Set3")
        self.logger.experiment.add_figure(f'{mode}/TSNE', fig , self.current_epoch)

    def visualize(self, signal, reonstructed_signal, mode):
        fig, axes = plt.subplots(5, 1, figsize=(15,6))
        for i in range(5):
            line1 = axes[i].plot(range(len(signal[i, :])), signal[i, :], color="tab:blue",  label="Real Signal")
            line2 = axes[i].plot(range(len(reonstructed_signal[i, :])), reonstructed_signal[i, :], color="tab:red", linestyle='dashed', label="Reconstructed Signal")
            axes[i].set_xticks([])
        fig.legend(handles =[line1[0], line2[0]], loc ='lower center', ncol=4)
        self.logger.experiment.add_figure(f'{mode}/Visualize', fig , self.current_epoch)

    def denormalize(self, signal, reonstructed_signal, min_max):
        n = signal.shape[0]
        output_signal = np.zeros_like(signal, dtype=float)
        output_reconstruct = np.zeros_like(signal, dtype=float)
        for i in range(n):
            for j in range(5):
                output_reconstruct[i, j, :] = reonstructed_signal[i, j, :] * (min_max[j][1] - min_max[j][0]) + min_max[j][0]
                output_signal[i, j, :] = signal[i, j, :] * (min_max[j][1] - min_max[j][0]) + min_max[j][0]

        return output_signal, output_reconstruct
    
    def display_prediction(self, target, prediction):
        prediction1 = np.argmax(softmax(prediction[:, 0, :], axis=1), axis=1)
        prediction2 = np.argmax(softmax(prediction[:, 1, :], axis=1), axis=1)
        prediction3 = np.argmax(softmax(prediction[:, 2, :], axis=1), axis=1)

        target1 = np.argmax(softmax(target[:, 0, :], axis=1), axis=1)
        target2 = np.argmax(softmax(target[:, 1, :], axis=1), axis=1)
        target3 = np.argmax(softmax(target[:, 2, :], axis=1), axis=1)

        df = pd.DataFrame({
            "No.7 Prediction (%)": prediction1*10, 
            "No.7 Target (%)": target1*10, 
            "No.22 Prediction (%)": prediction2*10, 
            "No.22 Target (%)": target2*10, 
            "No.38 Prediction (%)": prediction3*10, 
            "No.38 Target (%)": target3*10, 
        })
  
        df.to_csv(os.path.join(self.trainer.log_dir, "pred.csv"), index=False)
   