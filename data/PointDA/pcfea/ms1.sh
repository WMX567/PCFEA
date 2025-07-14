#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=32GB
#SBATCH --time=00:30:00
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --output=ms3ver13.out


source ~/.bashrc
conda activate py38


python /scratch/mw4355/PCFEA/train_PCFEA_cls.py --dataroot /scratch/mw4355/PCFEA/data/PointDA/ --src_dataset modelnet --trgt_dataset scannet --seed 3 --model_type ver13
