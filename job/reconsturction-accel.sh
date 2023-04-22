#!/bin/bash
#PBS -l select=1:ncpus=4:gpu_id=3
###PBS -l place=excl
#PBS -o out.txt				
#PBS -e err.txt				
#PBS -N reconstruction-accel
cd ~/ICSHM2022			

source ~/.bashrc			
conda activate icshm	

module load cuda-11.7			

python3 train_acceleration_reconstruction.py --arch reconstruction --trainer EncoderDecoder --load_model None --transfer False --max_epoch 500 --description From_scratch
python3 train_acceleration_reconstruction.py --arch reconstruction --trainer EncoderDecoder --load_model AE --transfer False --max_epoch 500 --description Pretrain_AE
python3 train_acceleration_reconstruction.py --arch reconstruction --trainer EncoderDecoder --load_model AE --transfer True --max_epoch 500 --description Transfer_AE