import os
import yaml
import argparse
from argparse import Namespace

from importlib import import_module

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from dataset.DataGeneration import DamageDataGenerationDataset


def create_dataloader(source):
    num_workers = min(os.cpu_count(), 4)
    train_dataset = DamageDataGenerationDataset(path="./Data", source=source)
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=num_workers) 

    return train_dataloader



def main(args):


    max_epochs = args.max_epoch
    model = import_module(f'model.{args.arch}').__dict__[args.trainer](args.mean_constraint)

    print(f"Total number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1024/1024:.4f} MB", )

    train_dataloader = create_dataloader(args.source)
    
    if args.mean_constraint:
        postfix = "mean_constraint"
    else:
        postfix = "no_constraint"

    #TensorBoard
    save_dir = f"Logs/Generation/Displacement_{postfix}"

    name = f"{args.description}/"

    # Save top-3 val loss models
    checkpoint_best_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="FJD", 
        mode="min",
        filename="{epoch:05d}-{fjd:.4f}"
    )

    # Save model at the middle epoch and last
    checkpoint_epoch_callback = ModelCheckpoint(
        every_n_epochs=int((max_epochs + 1)/2),
        filename="{epoch:05d}"
    )


    logger = TensorBoardLogger(
        save_dir = save_dir, 
        version = args.version,
        name = name,
        default_hp_metric = True)
    
    
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
                train_dataloaders=train_dataloader)
    
    
    # Save args
    with open(os.path.join(logger.log_dir, "args.yaml"), 'w') as outfile:
        yaml.dump(args, outfile, default_flow_style=False)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Quantization')

    parser.add_argument('--gpu', type=bool, default=True, help = 'Whether use GPU training')
    parser.add_argument('--device', type=int, default=1,  help = 'GPU id (If use the GPU)')
    parser.add_argument('--max_epoch', type=int, default=1000, help = 'Maximun epochs')

    parser.add_argument('--arch', type=str,  default="generation", help = 'The file where trainer located')
    parser.add_argument('--trainer', type=str,  default="WCGAN_GP", help = 'The trainer we used')

    parser.add_argument("--source", type=str, default="Displacement(16384)", help = 'The source of dataset.')

    parser.add_argument('--mean_constraint', action="store_true", default=False, help = 'Whether to apply the mean constraint')

    parser.add_argument('--description', type=str, default="None", help = 'description of the experiment')
    parser.add_argument('--version', type=int, help = 'version')

    args = parser.parse_args()

    main(args)