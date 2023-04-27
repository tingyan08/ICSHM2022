#!/bin/bash
#PBS -l select=1:ncpus=2:gpu_id=2
###PBS -l place=shared
#PBS -o out.txt				
#PBS -e err.txt				
#PBS -N reconstruction-disp
cd ~/ICSHM2022			

source ~/.bashrc			
conda activate icshm	

module load cuda-11.7			


# python3 train_reconstruction.py --arch reconstruction --trainer EncoderDecoder  --source Displacement --max_epoch 500 --description LAST
# python3 train_reconstruction.py --arch reconstruction --trainer EncoderDecoder  --source Displacement --pretrain --max_epoch 500 --description LAST
python3 train_reconstruction.py --arch reconstruction --trainer EncoderDecoder  --source Displacement --transfer --max_epoch 500 --description LAST