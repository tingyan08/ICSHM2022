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


from model.Displacement.extraction import AE, DamageAE, TripletAE
from ..utils import DoubleConv, Down, Up, OutConv


class Encoder(nn.Module):
    def __init__(self, bilinear = False):
        super(Encoder, self).__init__()
        self.input_conv = nn.Sequential(
            nn.Conv1d(5, 64, 3, 1, 1),
        )
        self.inc = (DoubleConv(64, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor) )

    def forward(self, x):
        x = self.input_conv(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x1, x2, x3, x4, x5
    
class Decoder(nn.Module):
    def __init__(self, bilinear = False):
        super(Decoder, self).__init__()
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, 64))
        self.output_conv = nn.Sequential(
            nn.Conv1d(64, 5, 3, 1, 1), 
        )

    def forward(self, x1, x2, x3, x4, x5):
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        logits = self.output_conv(x)
        return logits, x5
    
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

class EncoderDecoder(LightningModule):

    def __init__(self, load_model=None, transfer=False):
        super().__init__()
        if load_model != "None":
            if load_model == "DamageAE":
                self.encoder = DamageAE.load_from_checkpoint(
                "./Logs/Extraction/Displacement-DamageAE/Final/version_0/checkpoints/epoch=00424-train_loss=0.00303373.ckpt").to(self.device)
                if transfer:
                    self.encoder.freeze()
                self.encoder = self.encoder.encoder
                
            elif load_model == "TripletAE":
                self.encoder = TripletAE.load_from_checkpoint(
                "./Logs/Extraction/Displacement-TripletAE/Final/version_0/checkpoints/epoch=00472-train_loss=0.00460899.ckpt").to(self.device)
                if transfer:
                    self.encoder.freeze()
                self.encoder = self.encoder.encoder

            elif load_model == "AE":
                self.encoder = AE.load_from_checkpoint(
                "./Logs/Extraction/Displacement-AE/Final/version_0/checkpoints/epoch=00500-train_loss=0.00128296.ckpt").to(self.device)
                if transfer:
                    self.encoder.freeze()
                self.encoder = self.encoder.encoder

            else:
                raise Exception("Pretrianed model is not applied")
            

        else:
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
        masked_signal, target_signal = batch
        representation = self.encoder(masked_signal)
        prediction = self.decoder(representation)

        mse = nn.MSELoss()
        mse_loss  = mse(prediction, target_signal)
        loss =  mse_loss
        
        self.logger.experiment.add_scalar(f'Learning rate', self.optimizers().param_groups[0]['lr'], self.current_epoch)
        return {"loss": loss, "mse_loss": mse_loss}

    def validation_step(self, batch, batch_idx):
        masked_signal, target_signal = batch
        representation = self.encoder(masked_signal)
        prediction = self.decoder(representation)

        mse = nn.MSELoss()
        mse_loss  = mse(prediction, target_signal)
        loss =  mse_loss

        self.log("val_loss", loss)
        
        return {"loss": loss, "mse_loss": mse_loss, \
                "masked_signal": masked_signal, "prediction": prediction, "target_signal":target_signal}

        

    def configure_optimizers(self):
        lr = 5E-5

        opt = torch.optim.Adam(self.parameters(), lr=lr)
        return [opt], []
    
    

    def training_epoch_end(self, training_step_outputs):
        loss = []
        mse_loss = []

        for step_result in training_step_outputs:
            loss.append(step_result["loss"].cpu().detach().numpy())
            mse_loss.append(step_result["mse_loss"].cpu().detach().numpy())
            
        loss = np.concatenate([loss], axis=0)
        mse_loss = np.concatenate([mse_loss], axis=0)

        self.logger.experiment.add_scalar(f'Train/Loss/Loss', loss.mean(), self.current_epoch)
        self.logger.experiment.add_scalar(f'Train/Loss/MSE Loss', mse_loss.mean(), self.current_epoch)


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

        self.logger.experiment.add_scalar(f'Train/Loss/Loss', loss.mean(), self.current_epoch)
        self.logger.experiment.add_scalar(f'Train/Loss/MSE Loss', mse_loss.mean(), self.current_epoch)

        self.log("val_loss", mse_loss.mean())
    
        masked_signal = np.concatenate(masked_signal, axis=0)
        prediction = np.concatenate(prediction, axis=0)
        target_signal = np.concatenate(target_signal, axis=0)

        min_max = self.trainer.val_dataloaders[0].dataset.min_max
        prediction, target_signal = self.denormalize(prediction, target_signal, min_max)

        self.visualize_masked_process_reconstructions(masked_signal, prediction, target_signal, "A")
        self.visualize_masked_process_reconstructions(masked_signal, prediction, target_signal, "B")


    def denormalize(self, prediction, target_signal, min_max):
        n = prediction.shape[0]
        output_predcition = np.zeros_like(prediction, dtype=float)
        output_target = np.zeros_like(target_signal, dtype=float)
        for i in range(n):
            for j in range(5):
                output_predcition[i, j, :] = prediction[i, j, :] * (min_max[j][1] - min_max[j][0]) + min_max[j][0]
                output_target[i, j, :] = target_signal[i, j, :] * (min_max[j][1] - min_max[j][0]) + min_max[j][0]

        return output_predcition, output_target

    def visualize_masked_process_reconstructions(self, masked_signal, prediction, target_signal, task):

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


        plt_length = 512

        bs, num, length = target_signal.shape
        
        fig, axes = plt.subplots(num, 1, figsize=(20,8))
        for i in range(num):
            if len(np.unique(masked_signal[id, i, :plt_length])) != 1:
                line1 = axes[i].plot(range(len(target_signal[id, i, :plt_length])), target_signal[id, i, :plt_length], color="tab:orange",  label="Original Signal")
                line2 = axes[i].plot(range(len(prediction[id, i, :plt_length])), prediction[id, i, :plt_length], color="tab:green", linestyle="--",  label="Reconstruction Signal")          
            else:
                line3 = axes[i].plot(range(len(target_signal[id, i, :plt_length])), target_signal[id, i, :plt_length], color="tab:blue",  label="Original Signal (Masked)")
                line4 = axes[i].plot(range(len(prediction[id, i, :plt_length])), prediction[id, i, :plt_length], color="tab:red", linestyle="--",  label="Reconstruction Signal  (Masked)") 
            
            axes[i].set_xticks([])
        
        fig.suptitle(f"Epoch {self.current_epoch}")
        fig.legend(handles =[line1[0], line2[0], line3[0], line4[0]], loc ='lower center', ncol=4)
        self.logger.experiment.add_figure(f'Train/Visualize (Task {task})', fig , self.current_epoch)

