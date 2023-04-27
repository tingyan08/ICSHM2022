import os
import yaml
import argparse
import numpy as np
from argparse import Namespace

from importlib import import_module

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from dataset.SyntheticFinetune import SyntheticDataset

def get_all_condition():
    all_condition = []
    for d7 in range(6):
        for d22 in range(6):
            for d38 in range(6):
                onehot7 = np.eye(6)[d7]
                onehot22 = np.eye(6)[d22]
                onehot38 = np.eye(6)[d38]
                all_condition.append((onehot7, onehot22, onehot38))
    return all_condition


def create_dataloader(args):
    num_workers = min(os.cpu_count(), 4)
    train_dataset = SyntheticDataset(n_times=50, defined_condition=get_all_condition(), gan_checkpoint="./Logs/Generation/Displacement_WCGAN_GP/stride_dataset/version_0/checkpoints/epoch=00499.ckpt")
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    valid_dataset = SyntheticDataset(n_times=50, defined_condition=get_all_condition(), gan_checkpoint="./Logs/Generation/Displacement_WCGAN_GP/stride_dataset/version_0/checkpoints/epoch=00499.ckpt")
    valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

    return train_dataloader, valid_dataloader



def main(args):


    max_epochs = args.max_epoch
    model = import_module(f'model.Displacement.{args.arch}').__dict__[args.trainer]("./Logs/Identification/Displacement-regression/From_scratch/version_2/checkpoints/epoch=00343-val_loss=0.0000.ckpt")

    print("Total number of trainable parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    train_dataloader, valid_dataloader = create_dataloader(args)
    
    #TensorBoard
    save_dir = f"Logs/Identification/Displacement(FineTune)-{args.arch}"

    name = f"{args.description}/"


    logger = TensorBoardLogger(
        save_dir = save_dir, 
        version = args.version,
        name = name,
        default_hp_metric = True)
    
    
    if args.arch == "classification":
        # Save top-3 val loss models
        checkpoint_best_callback = ModelCheckpoint(
            save_top_k=1,
            monitor="val_acc", 
            mode="max",
            filename="{epoch:05d}-{val_acc:.4f}"
        )

    else:
        # Save top-3 val loss models
        checkpoint_best_callback = ModelCheckpoint(
            save_top_k=1,
            monitor="val_loss", 
            mode="min",
            filename="{epoch:05d}-{val_loss:.4f}"
        )


    # Save model at the middle epoch and last
    checkpoint_epoch_callback = ModelCheckpoint(
        every_n_epochs=int((max_epochs + 1)/2),
        filename="{epoch:05d}"
    )

    # training
    gpu = "gpu" if args.gpu else "cpu"
    trainer = Trainer(accelerator = gpu, devices = args.device,
                        logger = logger, log_every_n_steps = 1,
                        max_epochs = max_epochs + 1,
                        profiler = "simple", 
                        num_sanity_val_steps = 30,
                        callbacks = [checkpoint_best_callback, checkpoint_epoch_callback]
                        )


    trainer.fit(model,
                train_dataloaders=train_dataloader, 
                val_dataloaders=valid_dataloader)
    
    
    # Save args
    with open(os.path.join(logger.log_dir, "args.yaml"), 'w') as outfile:
        yaml.dump(args, outfile, default_flow_style=False)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Quantization')

    parser.add_argument('--gpu', type=bool, default=True, help = 'Whether use GPU training')
    parser.add_argument('--device', type=int, default=1,  help = 'GPU id (If use the GPU)')
    parser.add_argument('--max_epoch', type=int, default=1000, help = 'Maximun epochs')

    parser.add_argument('--arch', type=str,  default="regression", help = 'The file where trainer located')
    parser.add_argument('--trainer', type=str,  default="CNN_FineTune", help = 'The trainer we used')



    parser.add_argument('--description', type=str, default="None", help = 'description of the experiment')
    parser.add_argument('--version', type=int, help = 'version')

    args = parser.parse_args()

    main(args)