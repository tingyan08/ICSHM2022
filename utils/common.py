from __future__ import absolute_import
import datetime
import shutil
from pathlib import Path
import pdb
import json
import os

import torch
import logging
import functools

import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sn

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class checkpoint():
    def __init__(self, args):
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        today = datetime.date.today()

        self.args = args
        self.job_dir = Path(args.job_dir)
        self.ckpt_dir = self.job_dir / 'checkpoint'

        if args.reset:
            os.system('rm -rf ' + args.job_dir)

        def _make_dir(path):
            if not os.path.exists(path): os.makedirs(path)

        _make_dir(self.job_dir)
        _make_dir(self.ckpt_dir)

        
        config_dir = self.job_dir / 'config.json'
        with open(config_dir, 'w') as f:
            json.dump(vars(args), f, indent=4)
    
    def save_model(self, state, save_path):
        torch.save(state, os.path.join(self.ckpt_dir, save_path))
            
         
        

def get_logger(file_path):
    logger = logging.getLogger('gal')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt = '%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res



def plot_loss_lr(args, train_loss, valid_loss, lr):
    # Draw the loss curve
    fig = plt.figure()
    epoch = range(len(train_loss))
    plt.plot(epoch, train_loss, label="Training Loss")
    plt.plot(epoch, valid_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.legend()
    plt.savefig(f"{args.job_dir}/loss.png")
    plt.close(fig)

    # Draw the learning rate
    fig = plt.figure()
    plt.plot(epoch, lr, label="Learning rate")
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.savefig(f"{args.job_dir}/lr.png")
    plt.close(fig)


def plot_loss(args):
    df = pd.read_csv(f"{args.job_dir}/loss.csv")
    fig = plt.figure()
    plt.plot(df["Epoch"], df["training loss"], label="Training Loss")
    plt.plot(df["Epoch"], df["validation loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.legend()
    plt.savefig(f"{args.job_dir}/lossCurve.png")

def plot_acc(args):
    df = pd.read_csv(f"{args.job_dir}/loss.csv")
    fig = plt.figure()
    plt.plot(df["Epoch"], df["training acc"], label="Training Accuracy")
    plt.plot(df["Epoch"], df["validation acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy(%)")
    plt.legend()
    plt.savefig(f"{args.job_dir}/accCurve.png")

def plt_cm(args, metrics):
    cm = metrics.confmat.cpu().numpy()
    df_cm = pd.DataFrame(cm, index = [i for i in range(10)],
                  columns = [i for i in range(10)])
    plt.figure()
    sn.heatmap(df_cm, annot=True, cmap="Blues", fmt="d")
    plt.xlabel("Ground Truth")
    plt.ylabel("Prediction")
    plt.savefig(f"{args.job_dir}/confusionMatrix.png")


def plt_reconstruct(args, input, output, epoch, loss):
    limit = 1000
    if not os.path.exists(f"{args.job_dir}/validation"):
        os.makedirs(f"{args.job_dir}/validation")
    input = input.cpu().numpy()
    output = output.cpu().numpy()
    bs, ch, length = input.shape
    for i in range(bs):
        fig, ax = plt.subplots(ch, 1, figsize=(16, 6))
        for j in range(ch):
            ax[j].plot(range(limit), input[i][j][:limit], label="Original Signal")
            ax[j].plot(range(limit), output[i][j][:limit], "r--", label="Reconstructed Signal")
            ax[j].set_xticks([])
            
        fig.suptitle(f"Epoch {epoch+1} (MSE={loss})")
        fig.legend(["Original Signal", "Reconstructed Signal"], loc='lower center')
        plt.savefig(f"{args.job_dir}/validation/{epoch+1:04d}_{i:02d}.png")
        plt.close(fig)