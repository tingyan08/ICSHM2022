#!/bin/bash
#PBS -l select=1:ncpus=4:gpu_id=1
###PBS -l place=excl
#PBS -o out.txt				
#PBS -e err.txt				
#PBS -N generation-disp
cd ~/ICSHM2022			

source ~/.bashrc			
conda activate icshm	

module load cuda-11.7			



python3 train_generation.py --arch generation --trainer WCGAN_GP --mean_constraint --max_epoch 200 --description LAST
