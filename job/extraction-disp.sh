#!/bin/bash
#PBS -l select=1:ncpus=2:gpu_id=1
###PBS -l place=shared
#PBS -o out.txt				
#PBS -e err.txt				
#PBS -N Extraction-disp
cd ~/ICSHM2022			

source ~/.bashrc			
conda activate icshm	

module load cuda-11.7			


# python3 train_displacement_extraction.py --arch extraction --trainer AE --max_epoch 500 --description LAST
# python3 train_displacement_extraction.py --arch extraction --trainer DamageAE --max_epoch 500 --description LAST
python3 train_displacement_extraction.py --arch extraction --trainer TripletAE --max_epoch 500 --description LAST