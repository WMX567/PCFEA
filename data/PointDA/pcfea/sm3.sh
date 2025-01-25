#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=32GB
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=sm3pcfea.out




eval "$(conda shell.bash hook)"
conda activate pyCLGL




python /scratch1/mengxiwu/PCFEA/train_PCFEA_cls.py --dataroot /scratch1/mengxiwu/PCFEA/data/PointDA/ --src_dataset scannet --trgt_dataset modelnet --seed 3 --model_type pcfea
