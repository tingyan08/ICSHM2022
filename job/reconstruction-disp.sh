#!/bin/bash
#PBS -l select=1:ncpus=4:gpu_id=2
###PBS -l place=excl
#PBS -o out.txt				
#PBS -e err.txt				
#PBS -N reconstruction-disp
cd ~/ICSHM2022			

source ~/.bashrc			
conda activate icshm	

module load cuda-11.7			



python3 train_displacement_reconstruction.py --arch reconstruction --trainer EncoderDecoder --load_model None --transfer False --max_epoch 500 --description From_scratch
python3 train_displacement_reconstruction.py --arch reconstruction --trainer EncoderDecoder --load_model AE --transfer False --max_epoch 500 --description Pretrain_AE
python3 train_displacement_reconstruction.py --arch reconstruction --trainer EncoderDecoder --load_model AE --transfer True --max_epoch 500 --description Transfer_AE
python3 train_displacement_reconstruction.py --arch reconstruction --trainer EncoderDecoder --load_model DamageAE --transfer False --max_epoch 500 --description Pretrain_DamageAE
python3 train_displacement_reconstruction.py --arch reconstruction --trainer EncoderDecoder --load_model DamageAE --transfer True --max_epoch 500 --description Transfer_DamageAE
python3 train_displacement_reconstruction.py --arch reconstruction --trainer EncoderDecoder --load_model TripletAE --transfer False --max_epoch 500 --description Pretrain_TripletAE
python3 train_displacement_reconstruction.py --arch reconstruction --trainer EncoderDecoder --load_model TripletAE --transfer True --max_epoch 500 --description Transfer_TripletAE