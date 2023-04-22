#!/bin/bash
#PBS -l select=1:ncpus=4:gpu_id=3
###PBS -l place=excl
#PBS -o out.txt				
#PBS -e err.txt				
#PBS -N Extraction-accel
cd ~/ICSHM2022			

source ~/.bashrc			
conda activate icshm	

module load cuda-11.7			

python3 train_acceleration_extraction.py --arch autoencoder --trainer AE --max_epoch 500 --description Final
