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






class generator(nn.Module):
    def __init__(self, input_dim=5, output_ch=5):
        super(generator, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, 3, 1, 1),
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
            nn.Conv1d(512, 1024, 32, 1, 0)
        )


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
        x = torch.concat([noise, condition], axis=1)
        x = x.unsqueeze(axis=1)
        x = self.encoder(x)
        x = self.decoder(x)
        return x



class discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, condition_length=18):
        super(discriminator, self).__init__()
        self.condition_length = condition_length


        self.model = nn.Sequential(
            nn.Conv1d(5+ self.condition_length, 16, 3, 1, 1),
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
            nn.Conv1d(512, 1024, 32, 1, 0)
        )


        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )




    def forward(self, input, condition):
        condition = condition.unsqueeze(-1).repeat(1, 1, 1024)
        x = torch.concat([input, condition], axis=1)
        x = self.model(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


class WCGAN_GP(LightningModule):

    def __init__(self,
                 input_ch=5,
                 output_ch=5,
                 gp_weight=10,
                 condition_length=18):
        super().__init__()
        self.save_hyperparameters()
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.latent_dim = 1024-condition_length
        self.condtion_length = condition_length

        self.gp_weight = gp_weight

        self.generator = generator()
        self.discriminator = discriminator(condition_length=self.condtion_length)

        self.feature_extractor = DamageAE.load_from_checkpoint(
            "./Logs/Extraction/autoencoder_DamageAE/Final/version_0/checkpoints/epoch=00424-train_loss=0.00303373.ckpt").to(self.device)
        self.feature_extractor.eval()
        self.feature_extractor.freeze()



    def forward(self, noise, condition):
        return self.generator(noise, condition)

    def gradient_penalty(self, real_data, generated_data, condition):
        batch_size = real_data.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1)
        alpha = alpha.expand_as(real_data)
        alpha = alpha.to(self.device)
        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        interpolated = Variable(interpolated, requires_grad=True)
        interpolated = interpolated.to(self.device)

        # Calculate probability of interpolated examples
        prob_interpolated = self.discriminator(interpolated, condition)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).to(self.device),
                               create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()
    
    def mse_loss(self, input, target):
        return F.mse_loss(input, target)

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_data, condition, situation = batch

        z = torch.rand((condition.shape[0], self.latent_dim)).to(self.device)
        # train generator
        if optimizer_idx == 0:

        
            # generate signal
            generated_data = self(z, condition)

            # adversarial loss is binary cross-entropy
            d_generated = self.discriminator(generated_data, condition)
            g_loss = -d_generated.mean()

            # transform to latent space
            input_latent, _ = self.feature_extractor(real_data)
            synthetic_latent, _ = self.feature_extractor(generated_data)
            
            self.logger.experiment.add_scalar(f'Learning rate', self.optimizers()[0].param_groups[0]['lr'], self.current_epoch)
            return {"loss": g_loss,  "real_data": real_data, "generated_data": generated_data,\
                    "input_latent":input_latent, "synthetic_latent":synthetic_latent, "situation": situation}

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples

            generated_data = self(z, condition)

            d_real = self.discriminator(real_data, condition)
            d_generated = self.discriminator(generated_data, condition)

            gradient_penalty = self.gradient_penalty(real_data, generated_data, condition)

            d_loss = d_generated.mean() - d_real.mean() + gradient_penalty

            return {"loss": d_loss}
        

    def configure_optimizers(self):
        lr_g = 5E-6
        lr_d = 2E-5

        opt_g = torch.optim.AdamW(self.generator.parameters(), lr=lr_g)
        opt_d = torch.optim.AdamW(self.discriminator.parameters(), lr=lr_d)
        return [opt_g, opt_d], []
    
    
    

    def training_epoch_end(self, training_step_outputs):
        G_loss = []
        D_loss = []

        generated_data = []
        real_data = []

        input_latent = []
        synthetic_latent = []
        situation = []
        
        for step_result in training_step_outputs:
            G_loss.append(step_result[0]["loss"].cpu().detach().numpy())
            D_loss.append(step_result[1]["loss"].cpu().detach().numpy())

            generated_data.append(step_result[0]["generated_data"].cpu().detach().numpy())
            real_data.append(step_result[0]["real_data"].cpu().detach().numpy())

            input_latent.append(step_result[0]["input_latent"].cpu().detach().numpy())
            synthetic_latent.append(step_result[0]["synthetic_latent"].cpu().detach().numpy())
            situation.append(step_result[0]["situation"].cpu().detach().numpy())
            
        G_loss = np.concatenate([G_loss], axis=0)
        D_loss = np.concatenate([D_loss], axis=0)
        self.logger.experiment.add_scalar(f'Train/Loss/G_loss', G_loss.mean(), self.current_epoch)
        self.logger.experiment.add_scalar(f'Train/Loss/D_loss', D_loss.mean(), self.current_epoch)

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
        length = 512
        for i in range(5):
            line1 = axes[i].plot(range(len(generated_data_sample[i, :length])), generated_data_sample[i, :length], color="tab:red", linestyle='dashed', label="Generated Signal")
            line2 = axes[i].plot(range(len(real_data_sample[i, :length])), real_data_sample[i, :length], color="tab:blue",  label="Real Signal")
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