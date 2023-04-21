import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torch.autograd import grad as torch_grad

import numpy as np

from timm.scheduler.cosine_lr import CosineLRScheduler

from pytorch_lightning import LightningModule
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from model.autoencoder import DamageAE
from scipy import linalg




class Decoder(nn.Module):
    def __init__(self, latent_dim=512, condition_length=18):
        super(Decoder, self).__init__()

        self.fc = nn.Linear(latent_dim+condition_length, 1024)

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

    
        

    def forward(self, noise, condition):
        condition = condition.reshape(condition.shape[0], -1)
        x = torch.concat([noise, condition], axis=1).float()
        x = self.fc(x)
        x = x.view(x.shape[0], -1, 1)
        x = self.decoder(x)
        return x



class Encoder(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, latent_dim=512, condition_length=18):
        super(Encoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Conv1d(5+condition_length, 16, 3, 1, 1),
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



        self.fc_mu = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )

        self.fc_var = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )




    def forward(self, input, condition):
        condition = condition.reshape(condition.shape[0], -1)
        condition = condition.unsqueeze(-1).repeat(1, 1, 1024)
        x = torch.concat([input, condition], axis=1).float()
        x = self.decoder(x)
        x = x.view(x.shape[0], -1)
        z_mu = self.fc_mu(x)
        z_var = self.fc_var(x)
        return x, z_mu, z_var
    
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 18)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class CVAE(LightningModule):

    def __init__(self):
        super().__init__()
        self.save_hyperparameters()


        self.encoder = Encoder()
        self.decoder = Decoder()
        self.classifier = Classifier()

        self.feature_extractor = DamageAE.load_from_checkpoint(
            "./Logs/Extraction/autoencoder_DamageAE/Final/version_0/checkpoints/epoch=00424-train_loss=0.00303373.ckpt").to(self.device)
        self.feature_extractor.eval()
        self.feature_extractor.freeze()



    def forward(self, noise, condition):
        return self.decoder(noise, condition)
    

    def sensor_ce_loss(self, damage_pred, damage_target):
        loss = torch.zeros(1).to(self.device)
        ce = nn.CrossEntropyLoss()
        for i in range(3):
            loss += ce(damage_pred[:, i*6:(i+1)*6], damage_target[:, i*6:(i+1)*6])

        return loss


    
    def mse_kld(self, x, recon_x, logvar, mu):
        recon_func = nn.MSELoss()
        recon_func.size_average = True
        MSE = recon_func(recon_x, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return MSE, KLD
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def training_step(self, batch, batch_idx):
        real_data, condition, situation = batch

        # Encode signal
        latent, mu, logvar = self.encoder(real_data, condition)
        z = self.reparameterize(mu, logvar)
        pred = self.classifier(latent)

        generated_data = self.decoder(z, condition)
        ce_loss = self.sensor_ce_loss(pred, condition)
        mse_loss, KL_diver  = self.mse_kld(real_data, generated_data, mu, logvar)


        loss = mse_loss + KL_diver + ce_loss
        

        # transform to latent space
        input_latent, _ = self.feature_extractor(real_data)
        synthetic_latent, _ = self.feature_extractor(generated_data)

        
        self.logger.experiment.add_scalar(f'Learning rate', self.optimizers().param_groups[0]['lr'], self.current_epoch)
        return {"loss": loss, "mse_loss": mse_loss, "KL": KL_diver, "ce_loss": ce_loss,\
                "real_data": real_data, "generated_data": generated_data, \
                "input_latent":input_latent, "synthetic_latent":synthetic_latent,"situation": situation}

        

    def configure_optimizers(self):
        lr = 5E-5

        opt = torch.optim.Adam(self.parameters(), lr=lr)
        return [opt], []
    
    

    def training_epoch_end(self, training_step_outputs):
        loss = []
        mse_loss = []
        ce_loss = []
        kl_divergence = []

        generated_data = []
        real_data = []

        input_latent = []
        synthetic_latent = []
        situation = []
        
        for step_result in training_step_outputs:
            loss.append(step_result["loss"].cpu().detach().numpy())
            mse_loss.append(step_result["mse_loss"].cpu().detach().numpy())
            ce_loss.append(step_result["ce_loss"].cpu().detach().numpy())
            kl_divergence.append(step_result["KL"].cpu().detach().numpy())

            generated_data.append(step_result["generated_data"].cpu().detach().numpy())
            real_data.append(step_result["real_data"].cpu().detach().numpy())

            input_latent.append(step_result["input_latent"].cpu().detach().numpy())
            synthetic_latent.append(step_result["synthetic_latent"].cpu().detach().numpy())
            situation.append(step_result["situation"].cpu().detach().numpy())
            
        loss = np.concatenate([loss], axis=0)
        mse_loss = np.concatenate([mse_loss], axis=0)
        ce_loss = np.concatenate([ce_loss], axis=0)
        kl_divergence = np.concatenate([kl_divergence], axis=0)
        self.logger.experiment.add_scalar(f'Train/Loss/Total Loss', loss.mean(), self.current_epoch)
        self.logger.experiment.add_scalar(f'Train/Loss/MSE Loss', mse_loss.mean(), self.current_epoch)
        self.logger.experiment.add_scalar(f'Train/Loss/CE Loss', ce_loss.mean(), self.current_epoch)
        self.logger.experiment.add_scalar(f'Train/Loss/KL Divergence', kl_divergence.mean(), self.current_epoch)


        generated_data = np.concatenate(generated_data, axis=0)
        real_data = np.concatenate(real_data, axis=0)
        min_max = self.trainer.train_dataloader.dataset.datasets.min_max
        generated_data, real_data = self.denormalize(generated_data, real_data, min_max)
        self.visualize(generated_data[-1], real_data[-1])


        input_latent = np.concatenate(input_latent, axis=0)
        synthetic_latent = np.concatenate(synthetic_latent, axis=0)
        situation = np.concatenate(situation, axis=0)

        self.calculate_fid(synthetic_latent, input_latent, situation)
        self.draw_tsne(synthetic_latent, input_latent, situation)
        self.draw_pca(synthetic_latent, input_latent, situation)
    
    
    def draw_pca(self, synthetic_latent, input_latent, situation):
        synthetic_latent = np.reshape(synthetic_latent, (synthetic_latent.shape[0], -1))
        input_latent = np.reshape(input_latent, (input_latent.shape[0], -1))
        synthetic_embedded = PCA(n_components=2).fit_transform(synthetic_latent)
        real_embedded = PCA(n_components=2).fit_transform(input_latent)
        fig = plt.figure(figsize=(15,10))
        plt.scatter(synthetic_embedded[:, 0], synthetic_embedded[:, 1], c=situation,  marker="+", alpha=0.6, cmap="Set3")
        plt.scatter(real_embedded[:, 0], real_embedded[:, 1], c=situation, marker=".", alpha=0.6, cmap="Set3")
        self.logger.experiment.add_figure(f'Train/PCA', fig , self.current_epoch)
    
    def draw_tsne(self, synthetic_latent, input_latent, situation):
        synthetic_latent = np.reshape(synthetic_latent, (synthetic_latent.shape[0], -1))
        input_latent = np.reshape(input_latent, (input_latent.shape[0], -1))
        synthetic_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(synthetic_latent)
        real_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(input_latent)
        fig = plt.figure(figsize=(15,10))
        plt.scatter(synthetic_embedded[:, 0], synthetic_embedded[:, 1], c=situation,  marker="+", alpha=0.6, cmap="Set3")
        plt.scatter(real_embedded[:, 0], real_embedded[:, 1], c=situation, marker=".", alpha=0.6,  cmap="Set3")
        self.logger.experiment.add_figure(f'Train/TSNE', fig , self.current_epoch)

    def visualize(self, generated_data_sample, real_data_sample):
        fig, axes = plt.subplots(5, 1, figsize=(15,6))
        for i in range(5):
            line1 = axes[i].plot(range(len(real_data_sample[i, :])), real_data_sample[i, :], color="tab:blue",  label="Real Signal")
            line2 = axes[i].plot(range(len(generated_data_sample[i, :])), generated_data_sample[i, :], color="tab:red", linestyle='dashed', label="Generated Signal")
            axes[i].set_xticks([])
        fig.legend(handles =[line1[0], line2[0]], loc ='lower center', ncol=4)
        self.logger.experiment.add_figure(f'Train/Visualize', fig , self.current_epoch)

    def denormalize(self, generated_data, real_data, min_max):
        n = generated_data.shape[0]
        for i in range(n):
            for j in range(5):
                generated_data[i, j, :] = generated_data[i, j, :] * (min_max[j, 1] - min_max[j, 0]) + min_max[j, 0]
                real_data[i, j, :] = real_data[i, j, :] * (min_max[j, 1] - min_max[j, 0]) + min_max[j, 0]

        return generated_data, real_data
    
    
    
    def calculate_fid(self, synthetic_latent, input_latent, situation):
        input_latent = np.reshape(input_latent,(input_latent.shape[0], -1))
        synthetic_latent = np.reshape(synthetic_latent,(synthetic_latent.shape[0], -1))

        mu1, sigma1 = self.calculate_activation_statistics(input_latent)
        mu2, sigma2 = self.calculate_activation_statistics(synthetic_latent)

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        diff = mu1 - mu2

        eps = 1E-6
        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        fid = (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)
        self.log("FID", fid)
        self.logger.experiment.add_scalar(f'Train/Loss/FID', fid, self.current_epoch)

    def calculate_activation_statistics(self, act):
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma