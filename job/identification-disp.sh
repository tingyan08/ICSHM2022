#!/bin/bash
#PBS -l select=1:ncpus=2:gpu_id=6
###PBS -l place=shared
#PBS -o out.txt				
#PBS -e err.txt				
#PBS -N Identification-disp
cd ~/ICSHM2022			

source ~/.bashrc			
conda activate icshm	

module load cuda-11.7			


python3 train_identification.py --arch regression --trainer CNN --source Displacement --max_epoch 500 --description LAST

python3 train_identification.py --arch regression --trainer CNN --source Displacement_16384 --max_epoch 500 --description LAST




# python3 train_identification.py --arch regression --trainer CNN --transfer --max_epoch 500 --description LAST
# python3 train_identification.py --arch regression --trainer CNN --pretrain --max_epoch 500 --description LAST
