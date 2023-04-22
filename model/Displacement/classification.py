
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

from model.Displacement.autoencoder import AE, DamageAE, TripletAE


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
                "./Logs/Extraction/Displacement-DamageAE/Add_validation/version_0/checkpoints/epoch=00482-val_loss=0.00486894.ckpt").to(self.device)
                if transfer:
                    self.model.freeze()
                self.model = self.model.encoder
                
            elif load_model == "TripletAE":
                self.model = TripletAE.load_from_checkpoint(
                "./Logs/Extraction/Displacement-TripletAE/Add_validation/version_0/checkpoints/epoch=00498-val_loss=0.00802427.ckpt").to(self.device)
                if transfer:
                    self.model.freeze()
                self.model = self.model.encoder

            elif load_model == "AE":
                self.model = AE.load_from_checkpoint(
                "Logs/Extraction/Displacement-AE/Add_validation/version_0/checkpoints/epoch=00496-val_loss=0.00344358.ckpt").to(self.device)
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
        x = self.classifier(x)
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

    
