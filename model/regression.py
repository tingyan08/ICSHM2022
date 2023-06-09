import os
import torch
import torch.nn as  nn
import torch.nn.functional as F
import torch.optim as optim

from timm.scheduler.cosine_lr import CosineLRScheduler
from torchmetrics import Accuracy
from sklearn.metrics import accuracy_score

import numpy as np
from pytorch_lightning import LightningModule

import pandas as pd
from scipy.special import softmax


from model.extraction import AE
from model.utils import DoubleConv, Down, Bottleneck, ResNet


class Encoder(nn.Module):
    def __init__(self, length, bilinear = False):
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
        self.linear = nn.Linear(length // 64, 1)


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
    

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
        
    
        
        


class CNN(LightningModule):
    def __init__(self, length=16384):
        super(CNN, self).__init__()
        self.model = Encoder(length=length)
        self.classifier = Classifier()
        self.save_hyperparameters()
        

    def forward(self, x):
        x = self.model(x)
        x = self.classifier(x[-1])
        return x
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4, betas=[0.9, 0.999])
        return [optimizer], []
    
    
    # def configure_optimizers(self):
    #     optimizer = optim.Adam(self.parameters(), lr=1e-4, betas=[0.9, 0.999])
    #     # scheduler = CosineLRScheduler(optimizer, t_initial=self.trainer.max_epochs, \
    #     #                               warmup_t=int(self.trainer.max_epochs/10), warmup_lr_init=5e-6, warmup_prefix=True)
    #     scheduler = CosineLRScheduler(optimizer, t_initial=self.trainer.max_epochs, \
    #                                   warmup_t=0, warmup_lr_init=5e-6, warmup_prefix=True)
    #     return [optimizer], [scheduler]
    
    # def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
    #     scheduler.step(epoch=self.current_epoch)  # timm's scheduler need the epoch value
    #     self.logger.experiment.add_scalar(f'Learning rate', scheduler.optimizer.param_groups[0]['lr'], self.current_epoch)


    def training_step(self, batch, batch_idx):
        input, target = batch
        pred = self.forward(input)
        loss = nn.MSELoss()(pred, target)

        
        self.logger.experiment.add_scalar(f'Learning rate', self.optimizers().param_groups[0]['lr'], self.current_epoch)

        return {"loss": loss , "pred1":pred[:, 0], "pred2":pred[:, 1], "pred3": pred[:, 2], "target1":target[:, 0], "target2": target[:, 1], "target3":target[:, 2]}
    
    
    def validation_step(self, batch, batch_idx):        
        input, target = batch
        pred = self.forward(input)
        loss = nn.MSELoss()(pred, target)

        return {"loss": loss, "pred1":pred[:, 0], "pred2":pred[:, 1], "pred3": pred[:, 2], "target1":target[:, 0], "target2": target[:, 1], "target3":target[:, 2]}


    def training_epoch_end(self, training_step_outputs):
        loss = np.array([i["loss"].detach().cpu() for i in training_step_outputs])
        self.logger.experiment.add_scalar(f'Train/Loss/Total', loss.mean(), self.current_epoch)



        pred_no7 = [i["pred1"].detach().cpu().numpy() for i in training_step_outputs]
        pred_no7 = np.concatenate(pred_no7)
        target_no7 = [i["target1"].detach().cpu().numpy() for i in training_step_outputs]
        target_no7 = np.concatenate(target_no7)

        pred_no22 = [i["pred2"].detach().cpu().numpy() for i in training_step_outputs]
        pred_no22 = np.concatenate(pred_no22)
        target_no22 = [i["target2"].detach().cpu().numpy() for i in training_step_outputs]
        target_no22 = np.concatenate(target_no22)

        pred_no38 = [i["pred3"].detach().cpu().numpy() for i in training_step_outputs]
        pred_no38 = np.concatenate(pred_no38)
        target_no38 = [i["target3"].detach().cpu().numpy() for i in training_step_outputs]
        target_no38 = np.concatenate(target_no38)

        self.display_prediction(pred_no7, pred_no22, pred_no38, target_no7, target_no22, target_no38, "Train")



    def validation_epoch_end(self, valdiation_step_outputs):
        loss = np.array([i["loss"].cpu() for i in valdiation_step_outputs])
        self.logger.experiment.add_scalar(f'Validation/Loss/Total', loss.mean(), self.current_epoch)


        pred_no7 = [i["pred1"].cpu().numpy() for i in valdiation_step_outputs]
        pred_no7 = np.concatenate(pred_no7)
        target_no7 = [i["target1"].cpu().numpy() for i in valdiation_step_outputs]
        target_no7 = np.concatenate(target_no7)

        pred_no22 = [i["pred2"].cpu().numpy() for i in valdiation_step_outputs]
        pred_no22 = np.concatenate(pred_no22)
        target_no22 = [i["target2"].cpu().numpy() for i in valdiation_step_outputs]
        target_no22 = np.concatenate(target_no22)

        pred_no38 = [i["pred3"].cpu().numpy() for i in valdiation_step_outputs]
        pred_no38 = np.concatenate(pred_no38)
        target_no38 = [i["target3"].cpu().numpy() for i in valdiation_step_outputs]
        target_no38 = np.concatenate(target_no38)

        self.log("val_loss", loss.mean())


        self.display_prediction(pred_no7, pred_no22, pred_no38, target_no7, target_no22, target_no38, "Validation")

    
    def display_prediction(self, pred_no7, pred_no22, pred_no38, target_no7, target_no22, target_no38, mode):

            df = pd.DataFrame({
                "No.7 Prediction (%)": pred_no7*100, 
                "No.7 Target (%)": target_no7*100, 
                "No.22 Prediction (%)": pred_no22*100, 
                "No.22 Target (%)": target_no22*100, 
                "No.38 Prediction (%)": pred_no38*100, 
                "No.38 Target (%)": target_no38*100, 
            })
    
            df.to_csv(os.path.join(self.trainer.log_dir, f"{mode}.csv"), index=False)


class ResNet18(LightningModule):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.model = ResNet(Bottleneck, [2, 2, 2, 2], num_classes=3, num_channels=5)
        self.save_hyperparameters()
        

    def forward(self, x):
        x = self.model(x)
        return x
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4, betas=[0.9, 0.999])
        return [optimizer], []
    
    
    # def configure_optimizers(self):
    #     optimizer = optim.Adam(self.parameters(), lr=1e-4, betas=[0.9, 0.999])
    #     # scheduler = CosineLRScheduler(optimizer, t_initial=self.trainer.max_epochs, \
    #     #                               warmup_t=int(self.trainer.max_epochs/10), warmup_lr_init=5e-6, warmup_prefix=True)
    #     scheduler = CosineLRScheduler(optimizer, t_initial=self.trainer.max_epochs, \
    #                                   warmup_t=0, warmup_lr_init=5e-6, warmup_prefix=True)
    #     return [optimizer], [scheduler]
    
    # def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
    #     scheduler.step(epoch=self.current_epoch)  # timm's scheduler need the epoch value
    #     self.logger.experiment.add_scalar(f'Learning rate', scheduler.optimizer.param_groups[0]['lr'], self.current_epoch)


    def training_step(self, batch, batch_idx):
        input, target = batch
        pred = self.forward(input)
        loss = nn.MSELoss()(pred, target)

        
        self.logger.experiment.add_scalar(f'Learning rate', self.optimizers().param_groups[0]['lr'], self.current_epoch)

        return {"loss": loss , "pred1":pred[:, 0], "pred2":pred[:, 1], "pred3": pred[:, 2], "target1":target[:, 0], "target2": target[:, 1], "target3":target[:, 2]}
    
    
    def validation_step(self, batch, batch_idx):        
        input, target = batch
        pred = self.forward(input)
        loss = nn.MSELoss()(pred, target)

        return {"loss": loss, "pred1":pred[:, 0], "pred2":pred[:, 1], "pred3": pred[:, 2], "target1":target[:, 0], "target2": target[:, 1], "target3":target[:, 2]}


    def training_epoch_end(self, training_step_outputs):
        loss = np.array([i["loss"].detach().cpu() for i in training_step_outputs])
        self.logger.experiment.add_scalar(f'Train/Loss/Total', loss.mean(), self.current_epoch)



        pred_no7 = [i["pred1"].detach().cpu().numpy() for i in training_step_outputs]
        pred_no7 = np.concatenate(pred_no7)
        target_no7 = [i["target1"].detach().cpu().numpy() for i in training_step_outputs]
        target_no7 = np.concatenate(target_no7)

        pred_no22 = [i["pred2"].detach().cpu().numpy() for i in training_step_outputs]
        pred_no22 = np.concatenate(pred_no22)
        target_no22 = [i["target2"].detach().cpu().numpy() for i in training_step_outputs]
        target_no22 = np.concatenate(target_no22)

        pred_no38 = [i["pred3"].detach().cpu().numpy() for i in training_step_outputs]
        pred_no38 = np.concatenate(pred_no38)
        target_no38 = [i["target3"].detach().cpu().numpy() for i in training_step_outputs]
        target_no38 = np.concatenate(target_no38)

        self.display_prediction(pred_no7, pred_no22, pred_no38, target_no7, target_no22, target_no38, "Train")



    def validation_epoch_end(self, valdiation_step_outputs):
        loss = np.array([i["loss"].cpu() for i in valdiation_step_outputs])
        self.logger.experiment.add_scalar(f'Validation/Loss/Total', loss.mean(), self.current_epoch)


        pred_no7 = [i["pred1"].cpu().numpy() for i in valdiation_step_outputs]
        pred_no7 = np.concatenate(pred_no7)
        target_no7 = [i["target1"].cpu().numpy() for i in valdiation_step_outputs]
        target_no7 = np.concatenate(target_no7)

        pred_no22 = [i["pred2"].cpu().numpy() for i in valdiation_step_outputs]
        pred_no22 = np.concatenate(pred_no22)
        target_no22 = [i["target2"].cpu().numpy() for i in valdiation_step_outputs]
        target_no22 = np.concatenate(target_no22)

        pred_no38 = [i["pred3"].cpu().numpy() for i in valdiation_step_outputs]
        pred_no38 = np.concatenate(pred_no38)
        target_no38 = [i["target3"].cpu().numpy() for i in valdiation_step_outputs]
        target_no38 = np.concatenate(target_no38)

        self.log("val_loss", loss.mean())


        self.display_prediction(pred_no7, pred_no22, pred_no38, target_no7, target_no22, target_no38, "Validation")

    
    def display_prediction(self, pred_no7, pred_no22, pred_no38, target_no7, target_no22, target_no38, mode):

            df = pd.DataFrame({
                "No.7 Prediction (%)": pred_no7*100, 
                "No.7 Target (%)": target_no7*100, 
                "No.22 Prediction (%)": pred_no22*100, 
                "No.22 Target (%)": target_no22*100, 
                "No.38 Prediction (%)": pred_no38*100, 
                "No.38 Target (%)": target_no38*100, 
            })
    
            df.to_csv(os.path.join(self.trainer.log_dir, f"{mode}.csv"), index=False)



class CNN_FineTune(LightningModule):
    def __init__(self, checkpoint="./epoch=00343-val_loss=0.0000.ckpt"):
        super(CNN_FineTune, self).__init__()
        self.model = CNN.load_from_checkpoint(checkpoint)
        self.save_hyperparameters()
        

    def forward(self, x):
        x = self.model(x)
        return x
    
    # def configure_optimizers(self):
    #     optimizer = optim.Adam(self.parameters(), lr=1e-5, betas=[0.9, 0.999])
    #     return [optimizer], []
    
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4, betas=[0.9, 0.999])
        # scheduler = CosineLRScheduler(optimizer, t_initial=self.trainer.max_epochs, \
        #                               warmup_t=int(self.trainer.max_epochs/10), warmup_lr_init=5e-6, warmup_prefix=True)
        scheduler = CosineLRScheduler(optimizer, t_initial=self.trainer.max_epochs, \
                                      warmup_t=0, warmup_lr_init=5e-6, warmup_prefix=True)
        return [optimizer], [scheduler]
    
    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.step(epoch=self.current_epoch)  # timm's scheduler need the epoch value
        self.logger.experiment.add_scalar(f'Learning rate', scheduler.optimizer.param_groups[0]['lr'], self.current_epoch)


    def training_step(self, batch, batch_idx):
        input, target = batch
        pred = self.forward(input)
        loss = nn.MSELoss()(pred, target)

        
        self.logger.experiment.add_scalar(f'Learning rate', self.optimizers().param_groups[0]['lr'], self.current_epoch)

        return {"loss": loss , "pred1":pred[0], "pred2":pred[1], "pred3": pred[2], "target1":target[0], "target2": target[1], "target3":target[2]}
    
    
    def validation_step(self, batch, batch_idx):
        input, target = batch
        pred = self.forward(input)
        loss = nn.MSELoss()(pred, target)

        return {"loss": loss, "pred1":pred[0], "pred2":pred[1], "pred3": pred[2], "target1":target[0], "target2": target[1], "target3":target[2]}


    def training_epoch_end(self, training_step_outputs):
        loss = np.array([i["loss"].detach().cpu() for i in training_step_outputs])
        self.logger.experiment.add_scalar(f'Train/Loss/Total', loss.mean(), self.current_epoch)



        pred_no7 = [i["pred1"].detach().cpu().numpy() for i in training_step_outputs]
        pred_no7 = np.concatenate(pred_no7)
        target_no7 = [i["target1"].detach().cpu().numpy() for i in training_step_outputs]
        target_no7 = np.concatenate(target_no7)

        pred_no22 = [i["pred2"].detach().cpu().numpy() for i in training_step_outputs]
        pred_no22 = np.concatenate(pred_no22)
        target_no22 = [i["target2"].detach().cpu().numpy() for i in training_step_outputs]
        target_no22 = np.concatenate(target_no22)

        pred_no38 = [i["pred3"].detach().cpu().numpy() for i in training_step_outputs]
        pred_no38 = np.concatenate(pred_no38)
        target_no38 = [i["target3"].detach().cpu().numpy() for i in training_step_outputs]
        target_no38 = np.concatenate(target_no38)

        self.display_prediction(pred_no7, pred_no22, pred_no38, target_no7, target_no22, target_no38, "Train")



    def validation_epoch_end(self, valdiation_step_outputs):

        loss = np.array([i["loss"].cpu() for i in valdiation_step_outputs])
        self.logger.experiment.add_scalar(f'Validation/Loss/Total', loss.mean(), self.current_epoch)


        pred_no7 = [i["pred1"].cpu().numpy() for i in valdiation_step_outputs]
        pred_no7 = np.concatenate(pred_no7)
        target_no7 = [i["target1"].cpu().numpy() for i in valdiation_step_outputs]
        target_no7 = np.concatenate(target_no7)

        pred_no22 = [i["pred2"].cpu().numpy() for i in valdiation_step_outputs]
        pred_no22 = np.concatenate(pred_no22)
        target_no22 = [i["target2"].cpu().numpy() for i in valdiation_step_outputs]
        target_no22 = np.concatenate(target_no22)

        pred_no38 = [i["pred3"].cpu().numpy() for i in valdiation_step_outputs]
        pred_no38 = np.concatenate(pred_no38)
        target_no38 = [i["target3"].cpu().numpy() for i in valdiation_step_outputs]
        target_no38 = np.concatenate(target_no38)

        self.log("val_loss", loss.mean())


        self.display_prediction(pred_no7, pred_no22, pred_no38, target_no7, target_no22, target_no38, "Validation")

    
    def display_prediction(self, pred_no7, pred_no22, pred_no38, target_no7, target_no22, target_no38, mode):

            df = pd.DataFrame({
                "No.7 Prediction (%)": pred_no7*100, 
                "No.7 Target (%)": target_no7*100, 
                "No.22 Prediction (%)": pred_no22*100, 
                "No.22 Target (%)": target_no22*100, 
                "No.38 Prediction (%)": pred_no38*100, 
                "No.38 Target (%)": target_no38*100, 
            })
    
            df.to_csv(os.path.join(self.trainer.log_dir, f"{mode}.csv"), index=False)