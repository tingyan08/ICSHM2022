#!/bin/bash
#PBS -l select=1:ncpus=2:gpu_id=1
###PBS -l place=shared
#PBS -o out.txt				
#PBS -e err.txt				
#PBS -N reconstruction-accel
cd ~/ICSHM2022			

source ~/.bashrc			
conda activate icshm	

module load cuda-11.7			

# python3 train_reconstruction.py --arch reconstruction --trainer EncoderDecoder  --source Acceleration --max_epoch 500 --description LAST
# python3 train_reconstruction.py --arch reconstruction --trainer EncoderDecoder  --source Acceleration --pretrain --max_epoch 500 --description LAST
python3 train_reconstruction.py --arch reconstruction --trainer EncoderDecoder  --source Acceleration --transfer --max_epoch 500 --description LAST