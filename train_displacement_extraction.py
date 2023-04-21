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
from ..dataset.Displacement.FeatureExtraction import FeatureExtractionDataset


def create_dataloader(source):
    num_workers = min(os.cpu_count(), 4)
    train_dataset = FeatureExtractionDataset(path="./Data", source=source, mode="train")
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    valid_dataset = FeatureExtractionDataset(path="./Data", source=source, mode="valid")
    valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

    return train_dataloader, valid_dataloader



def main(args):


    max_epochs = args.max_epoch
    model = import_module(f'model.{args.arch}').__dict__[args.trainer]()

    print("Total number of trainable parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    train_dataloader, valid_dataloader = create_dataloader(args.dataset)
    
    #TensorBoard
    save_dir = f"Logs/Extraction/{args.trainer}-{args.dataset}"

    name = f"{args.description}/"


    logger = TensorBoardLogger(
        save_dir = save_dir, 
        version = args.version,
        name = name,
        default_hp_metric = True)
    
    
    # Save top-3 val loss models
    checkpoint_best_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss", 
        mode="min",
        filename="{epoch:05d}-{valid_loss:.8f}"
    )


    # training
    gpu = "gpu" if args.gpu else "cpu"
    trainer = Trainer(accelerator = gpu, devices = args.device,
                        logger = logger, log_every_n_steps = 1,
                        max_epochs = max_epochs + 1,
                        profiler = "simple", 
                        num_sanity_val_steps = 30,
                        callbacks = [checkpoint_best_callback]
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

    parser.add_argument('--arch', type=str,  default="autoencoder", help = 'The file where trainer located')
    parser.add_argument('--trainer', type=str,  default="AE", help = 'The trainer we used')

    parser.add_argument('--dataset', type=str,  default="Accleration", help = 'The trainer we used')

    parser.add_argument('--description', type=str,  default="None")

    parser.add_argument('--version', type=int, help = 'version')

    args = parser.parse_args()

    main(args)