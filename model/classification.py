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
            nn.Linear(512, 11)
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
        
    
        
        


class CNN(LightningModule):
    def __init__(self, length):
        super(CNN, self).__init__()
        self.model = Encoder(length=length)
        self.classifier = Classifier()
        self.save_hyperparameters()
        

    def forward(self, x):
        x = self.model(x)
        x = self.classifier(x[-1])
        return x
    
    # def configure_optimizers(self):
    #     optimizer = optim.Adam(self.parameters(), lr=1e-6, betas=[0.9, 0.999])
    #     return [optimizer], []
    
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4, betas=[0.9, 0.999])
        scheduler = CosineLRScheduler(optimizer, t_initial=self.trainer.max_epochs, \
                                      warmup_t=int(self.trainer.max_epochs/10), warmup_lr_init=5e-6, warmup_prefix=True)
        # scheduler = CosineLRScheduler(optimizer, t_initial=self.trainer.max_epochs, \
        #                               warmup_t=0, warmup_lr_init=5e-6, warmup_prefix=True)
        return [optimizer], [scheduler]
    
    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.step(epoch=self.current_epoch)  # timm's scheduler need the epoch value
        self.logger.experiment.add_scalar(f'Learning rate', scheduler.optimizer.param_groups[0]['lr'], self.current_epoch)


    def training_step(self, batch, batch_idx):
        input, target = batch
        pred = self.forward(input)
        loss = nn.CrossEntropyLoss()(pred, target.squeeze())

        softmax = nn.Softmax(dim=1)
        pred = torch.argmax(softmax(pred), axis=1)
        target = target.squeeze()

        
        self.logger.experiment.add_scalar(f'Learning rate', self.optimizers().param_groups[0]['lr'], self.current_epoch)

        return {"loss": loss , "pred":pred, "target":target}
    
    def validation_step(self, batch, batch_idx):
        input, target = batch
        pred = self.forward(input)
        loss = nn.CrossEntropyLoss()(pred, target.squeeze())

        softmax = nn.Softmax(dim=1)
        pred = torch.argmax(softmax(pred), axis=1)
        target = target.squeeze()

        return {"loss": loss , "pred":pred, "target":target}


    def training_epoch_end(self, training_step_outputs):
        loss = np.array([i["loss"].detach().cpu() for i in training_step_outputs])
        self.logger.experiment.add_scalar(f'Train/Loss', loss.mean(), self.current_epoch)

        pred = [i["pred"].detach().cpu().numpy() for i in training_step_outputs]
        pred = np.concatenate(pred)
        target = [i["target"].detach().cpu().numpy() for i in training_step_outputs]
        target = np.concatenate(target)
        acc = accuracy_score(pred, target)


        self.logger.experiment.add_scalar(f'Train/Accuracy', acc , self.current_epoch)


    def validation_epoch_end(self, valdiation_step_outputs):
        loss = np.array([i["loss"].detach().cpu() for i in valdiation_step_outputs])
        self.logger.experiment.add_scalar(f'Validation/Loss', loss.mean(), self.current_epoch)

        pred = [i["pred"].detach().cpu().numpy() for i in valdiation_step_outputs]
        pred = np.concatenate(pred)
        target = [i["target"].detach().cpu().numpy() for i in valdiation_step_outputs]
        target = np.concatenate(target)
        acc = accuracy_score(pred, target)

        self.log("val_acc", acc)


        self.logger.experiment.add_scalar(f'Validation/Accuracy', acc , self.current_epoch)




class ResNet18(LightningModule):
    def __init__(self, num_classes=216):
        super(ResNet18, self).__init__()
        self.model = ResNet(Bottleneck, [2, 2, 2, 2], num_classes=num_classes, num_channels=5)
        self.save_hyperparameters()
        

    def forward(self, x):
        x = self.model(x)
        return x
    
    # def configure_optimizers(self):
    #     optimizer = optim.SGD(self.parameters(), lr=1e-3, momentum=0.9)
    #     return [optimizer], []
    
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3, betas=[0.9, 0.999])
        scheduler = CosineLRScheduler(optimizer, t_initial=self.trainer.max_epochs, \
                                      warmup_t=int(self.trainer.max_epochs/10), warmup_lr_init=1e-5, warmup_prefix=True)
        # scheduler = CosineLRScheduler(optimizer, t_initial=self.trainer.max_epochs, \
        #                               warmup_t=0, warmup_lr_init=5e-6, warmup_prefix=True)
        return [optimizer], [scheduler]
    
    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.step(epoch=self.current_epoch)  # timm's scheduler need the epoch value
        self.logger.experiment.add_scalar(f'Learning rate', scheduler.optimizer.param_groups[0]['lr'], self.current_epoch)


    def training_step(self, batch, batch_idx):
        input, target = batch
        pred = self.forward(input)
        loss = nn.CrossEntropyLoss()(pred, target.squeeze())

        softmax = nn.Softmax(dim=1)
        pred = torch.argmax(softmax(pred), axis=1)
        target = target.squeeze()

        
        self.logger.experiment.add_scalar(f'Learning rate', self.optimizers().param_groups[0]['lr'], self.current_epoch)

        return {"loss": loss , "pred":pred, "target":target}
    
    def validation_step(self, batch, batch_idx):
        input, target = batch
        pred = self.forward(input)
        loss = nn.CrossEntropyLoss()(pred, target.squeeze())

        softmax = nn.Softmax(dim=1)
        pred = torch.argmax(softmax(pred), axis=1)
        target = target.squeeze()

        return {"loss": loss , "pred":pred, "target":target}


    def training_epoch_end(self, training_step_outputs):
        loss = np.array([i["loss"].detach().cpu() for i in training_step_outputs])
        self.logger.experiment.add_scalar(f'Train/Loss', loss.mean(), self.current_epoch)

        pred = [i["pred"].detach().cpu().numpy() for i in training_step_outputs]
        pred = np.concatenate(pred)
        target = [i["target"].detach().cpu().numpy() for i in training_step_outputs]
        target = np.concatenate(target)
        acc = accuracy_score(pred, target)


        self.logger.experiment.add_scalar(f'Train/Accuracy', acc , self.current_epoch)


    def validation_epoch_end(self, valdiation_step_outputs):
        loss = np.array([i["loss"].detach().cpu() for i in valdiation_step_outputs])
        

        pred = [i["pred"].detach().cpu().numpy() for i in valdiation_step_outputs]
        pred = np.concatenate(pred)
        target = [i["target"].detach().cpu().numpy() for i in valdiation_step_outputs]
        target = np.concatenate(target)
        acc = accuracy_score(pred, target)

        self.log("val_acc", acc)

        self.logger.experiment.add_scalar(f'Validation/Loss', loss.mean(), self.current_epoch)
        self.logger.experiment.add_scalar(f'Validation/Accuracy', acc , self.current_epoch)




class ResNet18_finetune(LightningModule):
    def __init__(self):
        super(ResNet18_finetune, self).__init__()
        state_dict = torch.load("./Logs/Identification/classification-ResNet18_finetune-Displacement/finetune_real_with_synthetic/version_0/checkpoints/epoch=00019-val_acc=0.5052.ckpt")["state_dict"]
        self.model = ResNet18(num_classes=216)
        self.load_state_dict(state_dict=state_dict)
        self.model = self.model.to(self.device)
        self.save_hyperparameters()
        

    def forward(self, x):
        x = self.model(x)
        return x
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-5)
        return [optimizer], []
    
    
    # def configure_optimizers(self):
    #     optimizer = optim.Adam(self.parameters(), lr=1e-3, betas=[0.9, 0.999])
    #     scheduler = CosineLRScheduler(optimizer, t_initial=self.trainer.max_epochs, \
    #                                   warmup_t=int(self.trainer.max_epochs/10), warmup_lr_init=1e-5, warmup_prefix=True)
    #     # scheduler = CosineLRScheduler(optimizer, t_initial=self.trainer.max_epochs, \
    #     #                               warmup_t=0, warmup_lr_init=5e-6, warmup_prefix=True)
    #     return [optimizer], [scheduler]
    
    # def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
    #     scheduler.step(epoch=self.current_epoch)  # timm's scheduler need the epoch value
    #     self.logger.experiment.add_scalar(f'Learning rate', scheduler.optimizer.param_groups[0]['lr'], self.current_epoch)


    def training_step(self, batch, batch_idx):
        input, target = batch
        pred = self.forward(input)
        loss = nn.CrossEntropyLoss()(pred, target.squeeze())

        softmax = nn.Softmax(dim=1)
        pred = torch.argmax(softmax(pred), axis=1)
        target = target.squeeze()

        
        self.logger.experiment.add_scalar(f'Learning rate', self.optimizers().param_groups[0]['lr'], self.current_epoch)

        return {"loss": loss , "pred":pred, "target":target}
    
    def validation_step(self, batch, batch_idx):
        input, target = batch
        pred = self.forward(input)
        loss = nn.CrossEntropyLoss()(pred, target.squeeze())

        softmax = nn.Softmax(dim=1)
        pred = torch.argmax(softmax(pred), axis=1)
        target = target.squeeze()

        return {"loss": loss , "pred":pred, "target":target}


    def training_epoch_end(self, training_step_outputs):
        loss = np.array([i["loss"].detach().cpu() for i in training_step_outputs])
        self.logger.experiment.add_scalar(f'Train/Loss', loss.mean(), self.current_epoch)

        pred = [i["pred"].detach().cpu().numpy() for i in training_step_outputs]
        pred = np.concatenate(pred)
        target = [i["target"].detach().cpu().numpy() for i in training_step_outputs]
        target = np.concatenate(target)
        acc = accuracy_score(pred, target)


        self.logger.experiment.add_scalar(f'Train/Accuracy', acc , self.current_epoch)


    def validation_epoch_end(self, valdiation_step_outputs):
        loss = np.array([i["loss"].detach().cpu() for i in valdiation_step_outputs])
        

        pred = [i["pred"].detach().cpu().numpy() for i in valdiation_step_outputs]
        pred = np.concatenate(pred)
        target = [i["target"].detach().cpu().numpy() for i in valdiation_step_outputs]
        target = np.concatenate(target)
        acc = accuracy_score(pred, target)

        self.log("val_acc", acc)

        self.logger.experiment.add_scalar(f'Validation/Loss', loss.mean(), self.current_epoch)
        self.logger.experiment.add_scalar(f'Validation/Accuracy', acc , self.current_epoch)

