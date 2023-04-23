
import torch
import torch.nn as  nn
import torch.nn.functional as F
import torch.optim as optim

from timm.scheduler.cosine_lr import CosineLRScheduler
from torchmetrics import Accuracy
from sklearn.metrics import accuracy_score

import numpy as np
from pytorch_lightning import LightningModule

from torchinfo import summary

from model.Displacement.extraction import AE, DamageAE, TripletAE
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
        return x
        
    
        
        


class CNN(LightningModule):
    def __init__(self, load_model=None, transfer=False):
        super(CNN, self).__init__()
        if load_model != "None":
            if load_model == "DamageAE":
                self.model = DamageAE.load_from_checkpoint(
                "./Logs/Extraction/Displacement-DamageAE/Final/version_0/checkpoints/epoch=00424-train_loss=0.00303373.ckpt").to(self.device)
                if transfer:
                    self.model.freeze()
                self.model = self.model.encoder
                
            elif load_model == "TripletAE":
                self.model = TripletAE.load_from_checkpoint(
                "./Logs/Extraction/Displacement-TripletAE/Final/version_0/checkpoints/epoch=00472-train_loss=0.00460899.ckpt").to(self.device)
                if transfer:
                    self.model.freeze()
                self.model = self.model.encoder

            elif load_model == "AE":
                self.model = AE.load_from_checkpoint(
                "./Logs/Extraction/Displacement-AE/Final/version_0/checkpoints/epoch=00500-train_loss=0.00128296.ckpt").to(self.device)
                if transfer:
                    self.model.freeze()
                self.model = self.model.encoder

            else:
                raise Exception("Pretrianed model is not applied")
            

        else:
            self.model = Encoder()
        self.classifier = Classifier()
        self.save_hyperparameters()
        

    def forward(self, x):
        x = self.model(x)
        x = self.classifier(x[-1])
        return x[:, :6], x[:, 6:12], x[:, 12:18]
    
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
        input, target1, target2, target3, signal_id = batch
        pred1, pred2, pred3 = self.forward(input)
        loss_no7 = nn.CrossEntropyLoss()(pred1, target1)
        loss_no22 = nn.CrossEntropyLoss()(pred2, target2)
        loss_no38 = nn.CrossEntropyLoss()(pred3, target3)
        total_loss = loss_no7 + loss_no22 + loss_no38

        softmax = nn.Softmax(dim=1)
        

        pred1 = torch.argmax(softmax(pred1), axis=1)
        pred2 = torch.argmax(softmax(pred2), axis=1)
        pred3 = torch.argmax(softmax(pred3), axis=1)

        target1 = torch.argmax(target1, axis=1)
        target2 = torch.argmax(target2, axis=1)
        target3 = torch.argmax(target3, axis=1)
        
        self.logger.experiment.add_scalar(f'Learning rate', self.optimizers().param_groups[0]['lr'], self.current_epoch)

        return {"loss": total_loss ,"loss_no7": loss_no7 , "loss_no22": loss_no22, "loss_no38": loss_no38, \
                "pred1":pred1, "pred2":pred2, "pred3": pred3, "target1":target1, "target2": target2, "target3":target3, \
                "signal_id":signal_id}
    
    
    def validation_step(self, batch, batch_idx):
        input, target1, target2, target3, signal_id = batch
        pred1, pred2, pred3 = self.forward(input)
        loss_no7 = nn.CrossEntropyLoss()(pred1, target1)
        loss_no22 = nn.CrossEntropyLoss()(pred2, target2)
        loss_no38 = nn.CrossEntropyLoss()(pred3, target3)
        total_loss = loss_no7 + loss_no22 + loss_no38

        softmax = nn.Softmax(dim=1)

        pred1 = torch.argmax(softmax(pred1), axis=1)
        pred2 = torch.argmax(softmax(pred2), axis=1)
        pred3 = torch.argmax(softmax(pred3), axis=1)

        target1 = torch.argmax(target1, axis=1)
        target2 = torch.argmax(target2, axis=1)
        target3 = torch.argmax(target3, axis=1)

        return {"loss": total_loss ,"loss_no7": loss_no7 , "loss_no22": loss_no22, "loss_no38": loss_no38, \
                "pred1":pred1, "pred2":pred2, "pred3": pred3, "target1":target1, "target2": target2, "target3":target3, \
                "signal_id":signal_id}


    def training_epoch_end(self, training_step_outputs):
        signal_id = [i["signal_id"].cpu().detach().numpy() for i in training_step_outputs]
        signal_id = np.concatenate(signal_id)

        loss = np.array([i["loss"].detach().cpu() for i in training_step_outputs])
        self.logger.experiment.add_scalar(f'Train/Loss/Total', loss.mean(), self.current_epoch)

        loss_no7 = np.array([i["loss_no7"].detach().cpu() for i in training_step_outputs])
        self.logger.experiment.add_scalar(f'Train/Loss/No.7', loss_no7.mean(), self.current_epoch)

        loss_no22 = np.array([i["loss_no22"].detach().cpu() for i in training_step_outputs])
        self.logger.experiment.add_scalar(f'Train/Loss/No.22', loss_no22.mean(), self.current_epoch)

        loss_no38 = np.array([i["loss_no38"].detach().cpu() for i in training_step_outputs])
        self.logger.experiment.add_scalar(f'Train/Loss/No.38', loss_no38.mean(), self.current_epoch)

        pred_no7 = [i["pred1"].detach().cpu().numpy() for i in training_step_outputs]
        pred_no7 = np.concatenate(pred_no7)
        target_no7 = [i["target1"].detach().cpu().numpy() for i in training_step_outputs]
        target_no7 = np.concatenate(target_no7)
        acc_no7 = accuracy_score(pred_no7, target_no7)

        pred_no22 = [i["pred2"].detach().cpu().numpy() for i in training_step_outputs]
        pred_no22 = np.concatenate(pred_no22)
        target_no22 = [i["target2"].detach().cpu().numpy() for i in training_step_outputs]
        target_no22 = np.concatenate(target_no22)
        acc_no22 = accuracy_score(pred_no22, target_no22)

        pred_no38 = [i["pred3"].detach().cpu().numpy() for i in training_step_outputs]
        pred_no38 = np.concatenate(pred_no38)
        target_no38 = [i["target3"].detach().cpu().numpy() for i in training_step_outputs]
        target_no38 = np.concatenate(target_no38)
        acc_no38 = accuracy_score(pred_no38, target_no38)

        self.logger.experiment.add_scalar(f'Train/Accuracy/Mean', (acc_no7 + acc_no22 + acc_no38) / 3 , self.current_epoch)
        self.logger.experiment.add_scalar(f'Train/Accuracy/No.7', acc_no7, self.current_epoch)
        self.logger.experiment.add_scalar(f'Train/Accuracy/No.22', acc_no22, self.current_epoch)
        self.logger.experiment.add_scalar(f'Train/Accuracy/No.38', acc_no38, self.current_epoch)


    def validation_epoch_end(self, valdiation_step_outputs):
        signal_id = [i["signal_id"].cpu().numpy() for i in valdiation_step_outputs]
        signal_id = np.concatenate(signal_id)

        loss = np.array([i["loss"].cpu() for i in valdiation_step_outputs])
        self.logger.experiment.add_scalar(f'Validation/Loss/Total', loss.mean(), self.current_epoch)

        loss_no7 = np.array([i["loss_no7"].cpu() for i in valdiation_step_outputs])
        self.logger.experiment.add_scalar(f'Validation/Loss/No.7', loss_no7.mean(), self.current_epoch)

        loss_no22 = np.array([i["loss_no22"].cpu() for i in valdiation_step_outputs])
        self.logger.experiment.add_scalar(f'Validation/Loss/No.22', loss_no22.mean(), self.current_epoch)

        loss_no38 = np.array([i["loss_no38"].cpu() for i in valdiation_step_outputs])
        self.logger.experiment.add_scalar(f'Validation/Loss/No.38', loss_no38.mean(), self.current_epoch)

        pred_no7 = [i["pred1"].cpu().numpy() for i in valdiation_step_outputs]
        pred_no7 = np.concatenate(pred_no7)
        target_no7 = [i["target1"].cpu().numpy() for i in valdiation_step_outputs]
        target_no7 = np.concatenate(target_no7)
        acc_no7 = accuracy_score(pred_no7, target_no7)

        pred_no22 = [i["pred2"].cpu().numpy() for i in valdiation_step_outputs]
        pred_no22 = np.concatenate(pred_no22)
        target_no22 = [i["target2"].cpu().numpy() for i in valdiation_step_outputs]
        target_no22 = np.concatenate(target_no22)
        acc_no22 = accuracy_score(pred_no22, target_no22)

        pred_no38 = [i["pred3"].cpu().numpy() for i in valdiation_step_outputs]
        pred_no38 = np.concatenate(pred_no38)
        target_no38 = [i["target3"].cpu().numpy() for i in valdiation_step_outputs]
        target_no38 = np.concatenate(target_no38)
        acc_no38 = accuracy_score(pred_no38, target_no38)

        mean_acc = (acc_no7 + acc_no22 + acc_no38) / 3
        self.log("val_acc", mean_acc)

        self.logger.experiment.add_scalar(f'Validation/Accuracy/Mean', (acc_no7 + acc_no22 + acc_no38) / 3 , self.current_epoch)
        self.logger.experiment.add_scalar(f'Validation/Accuracy/No.7', acc_no7, self.current_epoch)
        self.logger.experiment.add_scalar(f'Validation/Accuracy/No.22', acc_no22, self.current_epoch)
        self.logger.experiment.add_scalar(f'Validation/Accuracy/No.38', acc_no38, self.current_epoch)

    