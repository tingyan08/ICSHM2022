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


from model.extraction import AE
from model.utils import DoubleConv, Down, Up, OutConv



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

class EncoderDecoder(LightningModule):

    def __init__(self, source="Displacement", transfer=False, pretrain=False):
        super().__init__()
        if transfer :
            if source == "Displacement":
                self.AE = AE.load_from_checkpoint(
                    "./Logs/Extraction/Displacement/LAST/version_0/checkpoints/epoch=00197-val_loss=0.00000553.ckpt").to(self.device)
            else:
                self.AE = AE.load_from_checkpoint(
                    "./Logs/Extraction/Acceleration/LAST/version_0/checkpoints/epoch=00200-val_loss=0.00034191.ckpt").to(self.device)
                
            self.AE.freeze()
            self.encoder = self.AE.encoder
            self.decoder = Decoder()

        elif pretrain:
            if source == "Displacement":
                self.AE = AE.load_from_checkpoint(
                    "./Logs/Extraction/Displacement/LAST/version_0/checkpoints/epoch=00197-val_loss=0.00000553.ckpt").to(self.device)
            else:
                self.AE = AE.load_from_checkpoint(
                    "./Logs/Extraction/Acceleration/LAST/version_0/checkpoints/epoch=00200-val_loss=0.00034191.ckpt").to(self.device)
            self.encoder = self.AE.encoder
            self.decoder = self.AE.decoder

        else:
            self.encoder = Encoder()
            self.decoder = Decoder()


    def forward(self, X):
        latents = self.encoder(X)
        reconstruct = self.decoder(latents)
        return reconstruct


    def training_step(self, batch, batch_idx):
        masked_signal, target_signal = batch
        latents = self.encoder(masked_signal)
        prediction = self.decoder(latents)

        mse = nn.MSELoss()
        mse_loss  = mse(prediction, target_signal)
        loss =  mse_loss
        
        self.logger.experiment.add_scalar(f'Learning rate', self.optimizers().param_groups[0]['lr'], self.current_epoch)
        return {"loss": loss, "mse_loss": mse_loss, \
                "masked_signal": masked_signal, "prediction": prediction, "target_signal":target_signal}

    def validation_step(self, batch, batch_idx):
        masked_signal, target_signal = batch
        latents = self.encoder(masked_signal)
        prediction = self.decoder(latents)

        mse = nn.MSELoss()
        mse_loss  = mse(prediction, target_signal)
        loss =  mse_loss

        self.log("val_loss", loss)
        
        return {"loss": loss, "mse_loss": mse_loss, \
                "masked_signal": masked_signal, "prediction": prediction, "target_signal":target_signal}

        

    def configure_optimizers(self):
        lr = 5E-4

        opt = torch.optim.Adam(self.parameters(), lr=lr)
        return [opt], []
    
    

    def training_epoch_end(self, training_step_outputs):
        loss = []
        mse_loss = []

        masked_signal = []
        prediction = []
        target_signal = []

        for step_result in training_step_outputs:
            loss.append(step_result["loss"].cpu().detach().numpy())
            mse_loss.append(step_result["mse_loss"].cpu().detach().numpy())

            masked_signal.append(step_result["masked_signal"].cpu().detach().numpy())
            prediction.append(step_result["prediction"].cpu().detach().numpy())
            target_signal.append(step_result["target_signal"].cpu().detach().numpy())
            
        loss = np.concatenate([loss], axis=0)
        mse_loss = np.concatenate([mse_loss], axis=0)

        self.logger.experiment.add_scalar(f'Train/Loss/Loss', loss.mean(), self.current_epoch)
        self.logger.experiment.add_scalar(f'Train/Loss/MSE Loss', mse_loss.mean(), self.current_epoch)

        masked_signal = np.concatenate(masked_signal, axis=0)
        prediction = np.concatenate(prediction, axis=0)
        target_signal = np.concatenate(target_signal, axis=0)

        min_max = self.trainer.val_dataloaders[0].dataset.min_max
        prediction, target_signal = self.denormalize(prediction, target_signal, min_max)

        self.visualize_masked_process_reconstructions(masked_signal, prediction, target_signal, "A", "Validation")
        self.visualize_masked_process_reconstructions(masked_signal, prediction, target_signal, "B", "Validation")


    def validation_epoch_end(self, validation_step_outputs):
        loss = []
        mse_loss = []

        masked_signal = []
        prediction = []
        target_signal = []

        for step_result in validation_step_outputs:
            loss.append(step_result["loss"].cpu().detach().numpy())
            mse_loss.append(step_result["mse_loss"].cpu().detach().numpy())

            masked_signal.append(step_result["masked_signal"].cpu().detach().numpy())
            prediction.append(step_result["prediction"].cpu().detach().numpy())
            target_signal.append(step_result["target_signal"].cpu().detach().numpy())


            
        loss = np.concatenate([loss], axis=0)
        mse_loss = np.concatenate([mse_loss], axis=0)

        self.logger.experiment.add_scalar(f'Validation/Loss/Loss', loss.mean(), self.current_epoch)
        self.logger.experiment.add_scalar(f'Validation/Loss/MSE Loss', mse_loss.mean(), self.current_epoch)

        self.log("val_loss", mse_loss.mean())
    
        masked_signal = np.concatenate(masked_signal, axis=0)
        prediction = np.concatenate(prediction, axis=0)
        target_signal = np.concatenate(target_signal, axis=0)

        min_max = self.trainer.val_dataloaders[0].dataset.min_max
        prediction, target_signal = self.denormalize(prediction, target_signal, min_max)

        self.visualize_masked_process_reconstructions(masked_signal, prediction, target_signal, "A", "Validation")
        self.visualize_masked_process_reconstructions(masked_signal, prediction, target_signal, "B", "Validation")


    def denormalize(self, prediction, target_signal, min_max):
        n = prediction.shape[0]
        output_predcition = np.zeros_like(prediction, dtype=float)
        output_target = np.zeros_like(target_signal, dtype=float)
        for i in range(n):
            for j in range(5):
                output_predcition[i, j, :] = prediction[i, j, :] * (min_max[j][1] - min_max[j][0]) + min_max[j][0]
                output_target[i, j, :] = target_signal[i, j, :] * (min_max[j][1] - min_max[j][0]) + min_max[j][0]

        return output_predcition, output_target

    def visualize_masked_process_reconstructions(self, masked_signal, prediction, target_signal, task, mode):

        target = [1, 1, 1, 1, 0] if task == "A" else [1, 1, 0, 0, 0]

        for id, m in enumerate(masked_signal):
            mask = []
            for row in range(5):
                unique = np.unique(m[row, :])
                if len(unique) == 1 and unique[0] == 0:
                    mask.append(0)
                else:
                    mask.append(1)
            if target == mask:
                break

        bs, num, length = target_signal.shape
        
        fig, axes = plt.subplots(num, 1, figsize=(15,8))
        for i in range(num):
            if len(np.unique(masked_signal[id, i, :])) != 1:
                line1 = axes[i].plot(range(len(target_signal[id, i, :])), target_signal[id, i, :], color="tab:orange",  label="Original Signal")
                line2 = axes[i].plot(range(len(prediction[id, i, :])), prediction[id, i, :], color="tab:green", linestyle="--",  label="Reconstruction Signal")          
            else:
                line3 = axes[i].plot(range(len(target_signal[id, i, :])), target_signal[id, i, :], color="tab:blue",  label="Original Signal (Masked)")
                line4 = axes[i].plot(range(len(prediction[id, i, :])), prediction[id, i, :], color="tab:red", linestyle="--",  label="Reconstruction Signal  (Masked)") 
            
            axes[i].set_xticks([])
        
        fig.suptitle(f"Epoch {self.current_epoch}")
        fig.legend(handles =[line1[0], line2[0], line3[0], line4[0]], loc ='lower center', ncol=4)
        self.logger.experiment.add_figure(f'{mode}/Visualize (Task {task})', fig , self.current_epoch)

