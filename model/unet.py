import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from timm.scheduler.cosine_lr import CosineLRScheduler

from pytorch_lightning import LightningModule
import matplotlib.pyplot as plt




class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]


        x1 = F.pad(x1, [diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(LightningModule):
    def __init__(self, task="A", bilinear=False):
        super(UNet, self).__init__()
        self.task = task

        if self.task == "A":
            input_ch = 4
            output_ch = 1
        elif self.task == "B":
            input_ch = 2
            output_ch = 3
        else:
            input_ch = 5
            output_ch = 5


        self.input_conv = nn.Sequential(
            nn.Conv1d(input_ch, 64, 3, 1, 1),
        )

        self.inc = (DoubleConv(64, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, 64))

        self.output_conv = nn.Sequential(
            nn.Conv1d(64, output_ch, 3, 1, 1), 
        )
        

    def forward(self, x):
        x = self.input_conv(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        logits = self.output_conv(x)
        return logits, x5
    
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3, betas=[0.9, 0.999])
        scheduler = CosineLRScheduler(optimizer, t_initial=self.trainer.max_epochs, \
                                      warmup_t=int(self.trainer.max_epochs/10), warmup_lr_init=1e-5, warmup_prefix=True)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
    
    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.step(epoch=self.current_epoch)  # timm's scheduler need the epoch value
        self.logger.experiment.add_scalar(f'Learning rate', scheduler.optimizer.param_groups[0]['lr'], self.current_epoch)


    def training_step(self, batch, batch_idx):
        input, target = batch
        pred, latent = self.forward(input)
        loss_mse = nn.MSELoss()(pred, target)
        loss_mae = nn.L1Loss()(pred, target)   
        loss_rmse = torch.sqrt(nn.MSELoss()(pred, target))

        return {"loss": loss_mse,  "loss_mae": loss_mae.detach(), "loss_rmse": loss_rmse.detach(), "loss_mse": loss_mse.detach(),\
                "input": input, "target": target, "pred": pred}
    
    def validation_step(self, batch, batch_idx):
        input, target = batch
        pred, latent = self.forward(input)
        loss_mse = nn.MSELoss()(pred, target)
        loss_mae = nn.L1Loss()(pred, target)   
        loss_rmse = torch.sqrt(nn.MSELoss()(pred, target))

        self.log('val_loss', loss_mse)

        return {"loss": loss_mse,  "loss_mae": loss_mae.detach(), "loss_rmse": loss_rmse.detach(), "loss_mse": loss_mse.detach(),\
                "input": input, "target": target, "pred": pred}


    def training_epoch_end(self, training_step_outputs):
        mse = np.array([i["loss_mse"].cpu() for i in training_step_outputs])
        self.logger.experiment.add_scalar(f'Train/Loss/mse', mse.mean(), self.current_epoch)

        mae = np.array([i["loss_mse"].cpu() for i in training_step_outputs])
        self.logger.experiment.add_scalar(f'Train/Loss/mae', mae.mean(), self.current_epoch)

        rmse = np.array([i["loss_rmse"].cpu() for i in training_step_outputs])
        self.logger.experiment.add_scalar(f'Train/Loss/rmse', rmse.mean(), self.current_epoch)

        input = [i["input"].cpu().detach().numpy() for i in training_step_outputs]
        target = [i["target"].cpu().detach().numpy() for i in training_step_outputs]
        pred = [i["pred"].cpu().detach().numpy() for i in training_step_outputs]
        min_max = self.trainer.train_dataloader.dataset.datasets.min_max
        if self.task != "All":
            fig = self.visualize_process_reconstructions(input, target, pred, min_max, self.current_epoch)
            self.logger.experiment.add_figure(f'Train/Plot/Task {self.task}', fig , self.current_epoch)
        else:
            # Task A
            fig = self.visualize_masked_process_reconstructions(input, target, pred, min_max, self.current_epoch, "A")
            self.logger.experiment.add_figure('Train/Plot/Task A', fig , self.current_epoch)
            # Task B
            fig = self.visualize_masked_process_reconstructions(input, target, pred, min_max, self.current_epoch, "B")
            self.logger.experiment.add_figure('Train/Plot/Task B', fig , self.current_epoch)

    def validation_epoch_end(self, valdiation_step_outputs):
        mse = np.array([i["loss_mse"].cpu() for i in valdiation_step_outputs])
        self.logger.experiment.add_scalar(f'Validation/Loss/mse', mse.mean(), self.current_epoch)

        mae = np.array([i["loss_mse"].cpu() for i in valdiation_step_outputs])
        self.logger.experiment.add_scalar(f'Validation/Loss/mae', mae.mean(), self.current_epoch)

        rmse = np.array([i["loss_rmse"].cpu() for i in valdiation_step_outputs])
        self.logger.experiment.add_scalar(f'Validation/Loss/rmse', rmse.mean(), self.current_epoch)

        input = [i["input"].cpu().numpy() for i in valdiation_step_outputs]
        target = [i["target"].cpu().numpy() for i in valdiation_step_outputs]
        pred = [i["pred"].cpu().numpy() for i in valdiation_step_outputs]
        min_max = self.trainer.val_dataloaders[0].dataset.min_max
        if self.task != "All":
            fig = self.visualize_process_reconstructions(input, target, pred, min_max, self.current_epoch)
            self.logger.experiment.add_figure(f'Validation/Plot/Task {self.task}', fig , self.current_epoch)

        else:
            # Task A
            fig = self.visualize_masked_process_reconstructions(input, target, pred, min_max, self.current_epoch, "A")
            self.logger.experiment.add_figure('Validation/Plot/Task A', fig , self.current_epoch)
            # Task B
            fig = self.visualize_masked_process_reconstructions(input, target, pred, min_max, self.current_epoch, "B")
            self.logger.experiment.add_figure('Validation/Plot/Task B', fig , self.current_epoch)
        
    
    def visualize_process_reconstructions(self, input_list, target_list, pred_list, min_max, epoch):
        input = np.concatenate(input_list, axis=0)
        target = np.concatenate(target_list, axis=0)
        pred = np.concatenate(pred_list, axis=0)
        

        plt_length = 512

        bs, _, _ = input.shape
        fig, axes = plt.subplots(5, 1, figsize=(20,8))

        if self.task == "A":
            for i in range(4):
                line1 = axes[i].plot(range(len(input[0, i, :plt_length])), self.denormalize(input[0, i, :plt_length], min_max[i]), color="tab:green",  label="Input Signal")
                axes[i].set_xticks([])
                    
            line2 = axes[4].plot(range(len(target[0, 0, :plt_length])), self.denormalize(target[0, 0, :plt_length], min_max[4]), color="tab:blue",  label="Target Signal")          
            line3 = axes[4].plot(range(len(pred[0, 0, :plt_length])), self.denormalize(pred[0, 0, :plt_length], min_max[4]), color="tab:red", linestyle="--",  label="Predicted Signal")
            axes[4].set_xticks([]) 

        elif self.task == "B":
            for i in range(2):
                line1 = axes[i].plot(range(len(input[0, i, :plt_length])), self.denormalize(input[0, i, :plt_length], min_max[i]), color="tab:green",  label="Input Signal")
                axes[i].set_xticks([])
            for j in range(3):
                line2 = axes[j+2].plot(range(len(target[0, j, :plt_length])), self.denormalize(target[0, j, :plt_length], min_max[j+2]), color="tab:blue",  label="Target Signal")          
                line3 = axes[j+2].plot(range(len(pred[0, j, :plt_length])), self.denormalize(pred[0, j, :plt_length], min_max[j+2]), color="tab:red", linestyle="--",  label="Predicted Signal")
                axes[j].set_xticks([]) 
            
        
        fig.suptitle(f"Epoch {epoch+1}")
        fig.legend(handles =[line1[0], line2[0], line3[0]], loc ='lower center', ncol=4)
        return fig
    
    def visualize_masked_process_reconstructions(self, masked_input_list, original_input_list, pred_list, min_max, epoch, task):
        masked_input = np.concatenate(masked_input_list, axis=0)
        original_input = np.concatenate(original_input_list, axis=0)
        pred = np.concatenate(pred_list, axis=0)

        target = [1, 1, 1, 1, 0] if task == "A" else [1, 1, 0, 0, 0]

        for id, m in enumerate(masked_input):
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

        bs, num, length = original_input.shape
        
        fig, axes = plt.subplots(num, 1, figsize=(20,8))
        for i in range(num):
            if len(np.unique(masked_input[id, i, :plt_length])) != 1:
                line1 = axes[i].plot(range(len(original_input[id, i, :plt_length])), self.denormalize(original_input[id, i, :plt_length], min_max[i]), color="tab:orange",  label="Original Signal")
                line2 = axes[i].plot(range(len(pred[id, i, :plt_length])), self.denormalize(pred[id, i, :plt_length], min_max[i]), color="tab:green", linestyle="--",  label="Reconstruction Signal")          
            else:
                line3 = axes[i].plot(range(len(original_input[id, i, :plt_length])), self.denormalize(original_input[id, i, :plt_length], min_max[i]), color="tab:blue",  label="Original Signal (Masked)")
                line4 = axes[i].plot(range(len(pred[id, i, :plt_length])), self.denormalize(pred[id, i, :plt_length], min_max[i]), color="tab:red", linestyle="--",  label="Reconstruction Signal  (Masked)") 
            
            axes[i].set_xticks([])
        
        fig.suptitle(f"Epoch {epoch+1}")
        fig.legend(handles =[line1[0], line2[0], line3[0], line4[0]], loc ='lower center', ncol=4)
        return fig

    def denormalize(self, x, min_max):
        return x * (min_max[1] - min_max[0]) + min_max[0]

if __name__ == "__main__":
    model = UNet().to("cuda")
    input = torch.rand(1, 5, 10001).to("cuda")
    logits, latent = model(input)
    print(logits.shape)