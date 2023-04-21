import numpy as np
import torch
from torch import nn, optim
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class Encoder(nn.Module):
    """
    Encoder network containing enrolled LSTM/GRU
    :param number_of_features: number of input features
    :param hidden_size: hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: latent vector length
    :param dropout: percentage of nodes to dropout
    :param block: LSTM/GRU block
    """
    def __init__(self, number_of_features, condition_length, hidden_size, hidden_layer_depth, latent_length, dropout, block = 'LSTM'):

        super(Encoder, self).__init__()

        self.number_of_features = number_of_features
        self.hidden_size = hidden_size
        self.condition_length = condition_length
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length

        if block == 'LSTM':
            self.model = nn.LSTM(self.number_of_features, self.hidden_size, self.hidden_layer_depth, dropout = dropout)
        elif block == 'GRU':
            self.model = nn.GRU(self.number_of_features, self.hidden_size, self.hidden_layer_depth, dropout = dropout)
        else:
            raise NotImplementedError

    def forward(self, x):
        """Forward propagation of encoder. Given input, outputs the last hidden state of encoder
        :param x: input to the encoder, of shape (sequence_length, batch_size, number_of_features)
        :return: last hidden state of encoder, of shape (batch_size, hidden_size)
        """
        _, (h_end, c_end) = self.model(x)

        h_end = h_end[-1, :, :]
        return h_end


class Lambda(nn.Module):
    """Lambda module converts output of encoder to latent vector
    :param hidden_size: hidden size of the encoder
    :param latent_length: latent vector length
    """
    def __init__(self, hidden_size, condition_length, latent_length):
        super(Lambda, self).__init__()

        self.hidden_size = hidden_size
        self.latent_length = latent_length
        self.condition_length = condition_length

        self.hidden_to_mean = nn.Linear(self.hidden_size, self.latent_length)
        self.hidden_to_logvar = nn.Linear(self.hidden_size, self.latent_length)

        nn.init.xavier_uniform_(self.hidden_to_mean.weight)
        nn.init.xavier_uniform_(self.hidden_to_logvar.weight)

    def forward(self, cell_output):
        """Given last hidden state of encoder, passes through a linear layer, and finds the mean and variance
        :param cell_output: last hidden state of encoder
        :return: latent vector
        """
        self.latent_mean = self.hidden_to_mean(cell_output)
        self.latent_logvar = self.hidden_to_logvar(cell_output)

        if self.training:
            std = torch.exp(0.5 * self.latent_logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(self.latent_mean)
        else:
            return self.latent_mean

class Decoder(nn.Module):
    """Converts latent vector into output
    :param sequence_length: length of the input sequence
    :param batch_size: batch size of the input sequence
    :param hidden_size: hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: latent vector length
    :param output_size: 2, one representing the mean, other log std dev of the output
    :param block: GRU/LSTM - use the same which you've used in the encoder
    :param dtype: Depending on cuda enabled/disabled, create the tensor
    """
    def __init__(self, sequence_length, batch_size, hidden_size, hidden_layer_depth, latent_length, output_size, block='LSTM'):

        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length
        self.output_size = output_size

        if block == 'LSTM':
            self.model = nn.LSTM(1, self.hidden_size, self.hidden_layer_depth)
        elif block == 'GRU':
            self.model = nn.GRU(1, self.hidden_size, self.hidden_layer_depth)
        else:
            raise NotImplementedError

        self.latent_to_hidden = nn.Linear(self.latent_length, self.hidden_size)
        self.hidden_to_output = nn.Linear(self.hidden_size, self.output_size)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        

        nn.init.xavier_uniform_(self.latent_to_hidden.weight)
        nn.init.xavier_uniform_(self.hidden_to_output.weight)

    def forward(self, latent):
        """Converts latent to hidden to output
        :param latent: latent vector
        :return: outputs consisting of mean and std dev of vector
        """
        h_state = self.latent_to_hidden(latent)
        batch_size = h_state.shape[0]

        self.decoder_inputs = torch.zeros(self.sequence_length, batch_size, 1, requires_grad=True).to(self.device)
        self.c_0 = torch.zeros(self.hidden_layer_depth, batch_size, self.hidden_size, requires_grad=True).to(self.device)

        if isinstance(self.model, nn.LSTM):
            
            h_0 = torch.stack([h_state for _ in range(self.hidden_layer_depth)])
            decoder_output, _ = self.model(self.decoder_inputs, (h_0, self.c_0))
        elif isinstance(self.model, nn.GRU):
            h_0 = torch.stack([h_state for _ in range(self.hidden_layer_depth)])
            decoder_output, _ = self.model(self.decoder_inputs, h_0)
        else:
            raise NotImplementedError

        out = self.hidden_to_output(decoder_output)
        return out

    

class CVRAE(LightningModule):

    def __init__(self, sequence_length=200000, batch_size=32, number_of_features=5, condition_length=18, hidden_size=90, hidden_layer_depth=3, latent_length=100,
                 block='LSTM', dropout_rate=0.):

        super(CVRAE, self).__init__()


        self.encoder = Encoder(number_of_features = number_of_features,
                               hidden_size=hidden_size,
                               hidden_layer_depth=hidden_layer_depth,
                               latent_length=latent_length,
                               condition_length=condition_length,
                               dropout=dropout_rate,
                               block=block)

        self.lmbd = Lambda(hidden_size=hidden_size,
                           condition_length=condition_length,
                           latent_length=latent_length)

        self.decoder = Decoder(sequence_length=sequence_length,
                               batch_size=batch_size,
                               hidden_size=hidden_size,
                               hidden_layer_depth=hidden_layer_depth,
                               latent_length=latent_length,
                               output_size=number_of_features,
                               block=block)

        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length
        self.batch_size = batch_size

        

        self.save_hyperparameters()
       

    def forward(self, x, condition):
        cell_output = self.encoder(x)
        latent = self.lmbd(cell_output)
        x_decoded = self.decoder(latent)

        return x_decoded, latent

    

    def compute_loss(self, X, condition):
        x = Variable(X[:,:,:].type(self.dtype), requires_grad = True)
        loss_fn = nn.MSELoss(reduction='sum')

        x_decoded, _ = self(x, condition)
        latent_mean, latent_logvar = self.lmbd.latent_mean, self.lmbd.latent_logvar

        kl_loss = -0.5 * torch.mean(1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp())
        recon_loss = loss_fn(x_decoded, x)

        return recon_loss + kl_loss, recon_loss, kl_loss, x_decoded

    

    def training_step(self, batch, batch_idx):
        input, condition, situation = batch
        input = torch.permute(input, (2, 0, 1)) # B, H, L -> L, B, H

        loss, recon_loss, kl_loss, x_decoded = self.compute_loss(input, condition)


        
        self.logger.experiment.add_scalar(f'Learning rate', self.optimizers().param_groups[0]['lr'], self.current_epoch)
        return {"loss": loss, "mse": recon_loss, "KL": kl_loss, \
                "real": torch.permute(input, (1, 2, 0)), "synthetic": torch.permute(x_decoded, (1, 2, 0)), "situation": situation}

        

    def configure_optimizers(self):
        lr = 5E-4

        opt = torch.optim.AdamW(self.parameters(), lr=lr)
        return [opt], []
    
    

    def training_epoch_end(self, training_step_outputs):
        loss = []
        bce_loss = []
        kl_divergence = []
        generated_data = []
        real_data = []
        situation = []
        
        for step_result in training_step_outputs:
            loss.append(step_result["loss"].cpu().detach().numpy())
            bce_loss.append(step_result["mse"].cpu().detach().numpy())
            kl_divergence.append(step_result["KL"].cpu().detach().numpy())
            generated_data.append(step_result["synthetic"].cpu().detach().numpy())
            real_data.append(step_result["real"].cpu().detach().numpy())
            situation.append(step_result["situation"].cpu().detach().numpy())
            
        loss = np.concatenate([loss], axis=0)
        bce_loss = np.concatenate([bce_loss], axis=0)
        kl_divergence = np.concatenate([kl_divergence], axis=0)
        generated_data = np.concatenate(generated_data, axis=0)
        real_data = np.concatenate(real_data, axis=0)
        situation = np.concatenate(situation, axis=0)

        self.logger.experiment.add_scalar(f'Train/Loss/Total Loss', loss.mean(), self.current_epoch)
        self.logger.experiment.add_scalar(f'Train/Loss/BCE Loss', bce_loss.mean(), self.current_epoch)
        self.logger.experiment.add_scalar(f'Train/Loss/KL Divergence', kl_divergence.mean(), self.current_epoch)

        min_max = self.trainer.train_dataloader.dataset.datasets.min_max
        generated_data, real_data = self.denormalize(generated_data, real_data, min_max)

        
        self.visualize(generated_data[-1], real_data[-1])

        if self.current_epoch % 10 == 0:
            # generated_data, real_data, situation = self.sample(generated_data, real_data, situation, 0.3)
            self.draw_tsne(generated_data, real_data, situation)
            self.draw_pca(generated_data, real_data, situation)
    
    
    def draw_pca(self, generated_data, real_data, situation):
        generated_data = np.reshape(generated_data, (generated_data.shape[0], -1))
        real_data = np.reshape(real_data, (real_data.shape[0], -1))
        synthetic_embedded = PCA(n_components=2).fit_transform(generated_data)
        real_embedded = PCA(n_components=2).fit_transform(real_data)
        fig = plt.figure(figsize=(15,10))
        plt.scatter(synthetic_embedded[:, 0], synthetic_embedded[:, 1], c=situation,  marker="+", cmap="Set3")
        plt.scatter(real_embedded[:, 0], real_embedded[:, 1], c=situation, marker=".", cmap="Set3")
        self.logger.experiment.add_figure(f'Train/PCA', fig , self.current_epoch)
    
    def draw_tsne(self, generated_data, real_data, situation):
        generated_data = np.reshape(generated_data, (generated_data.shape[0], -1))
        real_data = np.reshape(real_data, (real_data.shape[0], -1))
        synthetic_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=10).fit_transform(generated_data)
        real_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=10).fit_transform(real_data)
        fig = plt.figure(figsize=(15,10))
        plt.scatter(synthetic_embedded[:, 0], synthetic_embedded[:, 1], c=situation,  marker="+", cmap="Set3")
        plt.scatter(real_embedded[:, 0], real_embedded[:, 1], c=situation, marker=".", cmap="Set3")
        self.logger.experiment.add_figure(f'Train/TSNE', fig , self.current_epoch)

    def visualize(self, generated_data_sample, real_data_sample):
        fig, axes = plt.subplots(5, 1, figsize=(15,6))
        for i in range(5):
            line1 = axes[i].plot(range(len(generated_data_sample[i, :])), generated_data_sample[i, :], color="tab:red", linestyle='dashed', label="Generated Signal")
            line2 = axes[i].plot(range(len(real_data_sample[i, :])), real_data_sample[i, :], color="tab:blue",  label="Real Signal")
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
    
    def sample(self, generated_data, real_data, situation, ratio):
        idx = np.random.randint(generated_data.shape[0], size=int(generated_data.shape[0] * ratio))
        return generated_data[idx, :], real_data[idx, :], situation[idx]