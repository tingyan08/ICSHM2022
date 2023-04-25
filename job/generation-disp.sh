#!/bin/bash
#PBS -l select=1:ncpus=4:gpu_id=2
###PBS -l place=excl
#PBS -o out.txt				
#PBS -e err.txt				
#PBS -N generation-disp
cd ~/ICSHM2022			

source ~/.bashrc			
conda activate icshm	

module load cuda-11.7			



python3 train_displacement_generation.py --arch generation --trainer WCGAN_GP --max_epoch 500 --description stride_dataset
