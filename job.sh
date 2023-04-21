#!/bin/bash
#PBS -l select=1:ncpus=4:gpu_id=2
###PBS -l place=excl
#PBS -o out.txt				
#PBS -e err.txt				
#PBS -N identification
cd ~/ICSHM/Damage_identification				

source ~/.bashrc			
conda activate icshm	

module load cuda-11.7			


# python3 train_generation.py --arch cvae_paper --trainer CVAE --data_type 1D --max_epoch 1000 --description add_classfier
# python3 train_reconstruction.py --arch unet --trainer UNet --task All --max_epoch 1000 
# python3 train_extraction.py --arch triplet --trainer TripletDamageAE --max_epoch 500 --description Final
python3 train_identification.py --arch classification --trainer CNN --load_model None --transfer False --max_epoch 500 --description From_scratch
python3 train_identification.py --arch classification --trainer CNN --load_model DamageAE --transfer False --max_epoch 500 --description Pretrain_DamageAE
python3 train_identification.py --arch classification --trainer CNN --load_model DamageAE --transfer True --max_epoch 500 --description Transfer_DamageAE
python3 train_identification.py --arch classification --trainer CNN --load_model TripletAE --transfer False --max_epoch 500 --description Pretrain_TripletAE
python3 train_identification.py --arch classification --trainer CNN --load_model TripletAE --transfer True --max_epoch 500 --description Transfer_TripletAE