import os
import pandas as pd

from arguments import args
import utils.common as utils

from importlib import import_module

import torch
import torch.nn as nn
import torch.optim as optim
from utils.scheduler import ExponetialLRScheduler

from torch.utils.data import DataLoader
from dataset.DamageIdentification import DamageIdentificationDataset
from tqdm import tqdm

import warnings, math

warnings.filterwarnings("ignore")

device = torch.device(f"cuda:{args.gpus[0]}")

checkpoint = utils.checkpoint(args)


def main():

    start_epoch = 0
    best_acc = 0.0
 
    # Data loading
    print('=> Preparing data..')
 
    # data loader
    
    train_dataset = DamageIdentificationDataset(path="./Data", mode="train")
    train_dataloader = DataLoader(train_dataset, batch_size=11, shuffle=True)

    valid_dataset = DamageIdentificationDataset(path="./Data", mode="valid")
    valid_dataloader = DataLoader(valid_dataset, batch_size=11, shuffle=True)


    
    
    
    # Create model
    print('=> Building model...')

    # load training model
    model = import_module(f'model.{args.arch}').__dict__[args.model]().to(device)
    

    # Check device
    print(f"=> Using divice: {device}")

    # Load pretrained weights
    if args.pretrained:
 
        ckpt = torch.load(os.path.join(checkpoint.ckpt_dir, args.source_file), map_location = device)
        state_dict = ckpt['state_dict']

        model.load_state_dict(state_dict)
        model = model.to(device)
        

    param = [param for name, param in model.named_parameters()]
    
    optimizer = optim.Adam(param, lr = args.init_lr, weight_decay = args.weight_decay)
    scheduler = ExponetialLRScheduler(optimizer, args.init_lr, args.peak_lr, args.final_lr, args.warmup_steps, args.num_epochs)


    # Step of svaing checkpoints
    save_step = args.num_epochs // 2

    with open(f"{args.job_dir}/loss.csv", "w") as f:
        f.writelines(f"Epoch,lr,training loss,validation loss\n")
        best_loss = 10000
        learning_rate = []
        train_loss_list = []
        val_loss_list = []



        for epoch in tqdm(range(start_epoch, args.num_epochs)):
            # Calculate and record the leraning rate
            lr = scheduler.step()
            learning_rate.append(lr)


            train_loss = train(args, train_dataloader, model, optimizer, epoch)
            valid_loss = valid(args, valid_dataloader, model)

            train_loss_list.append(train_loss)
            val_loss_list.append(valid_loss)

            f.writelines(f"{epoch},{scheduler.get_lr():.8f},{train_loss:.8f},{valid_loss:.8f}\n")
    
            is_best = best_loss < valid_loss
            best_loss = min(best_loss, valid_loss)
            

            state = {
                'state_dict': model.state_dict(),
                
                # 'optimizer': optimizer.state_dict(),
                
                # 'scheduler': scheduler.state_dict(),
                
                'epoch': epoch + 1
            }

            if ((epoch+1) % save_step == 0):
                checkpoint.save_model(state, f"{epoch+1}.pth")
            if is_best:
                checkpoint.save_model(state, f"best.pth")



        
    utils.plot_loss_lr(args, train_loss=train_loss_list, valid_loss=val_loss_list, lr=learning_rate)
    
    print(f'Best MES loss: {best_loss:.3f}\n')


  
       
def train(args, data_loader, model, optimizer, epoch):
    losses = utils.AverageMeter()

    criterion = nn.MSELoss()
    
    num_iterations = len(data_loader)
    
    # switch to train mode
    model.train()


    for inputs, targets in data_loader:
        
        # num_iters = num_iterations * epoch + i

        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # train
        output = model(inputs)
        loss = criterion(output, targets)

        # optimize cnn
        loss.backward()
        optimizer.step()

        ## train weights        
        losses.update(loss.item())
        

    return losses.sum

        # if i % args.print_freq == 0:     
        #     print(
        #         'Epoch[{0}]({1}/{2}): \n'
        #         'Train_loss: {train_loss.val:.4f} ({train_loss.avg:.4f})\n'
        #         'Train acc {acc.val:.3f} ({acc.avg:.3f})\n'.format(
        #         epoch, i, num_iterations, 
        #         train_loss = losses,
        #         acc = acc))
                
      
 
def valid(args, loader_valid, model):
    losses = utils.AverageMeter()

    criterion = nn.MSELoss()


    # switch to eval mode
    model.eval()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader_valid, 1):
            
            inputs = inputs.to(device)
            targets = targets.to(device)
             
            preds = model(inputs)
            loss = criterion(preds, targets)
        
            # image classification results
            losses.update(loss.item())


    return losses.sum
    


if __name__ == '__main__':
    main()

