#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=32GB
#SBATCH --time=22:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=ms+1pcfea.out




eval "$(conda shell.bash hook)"
conda activate pyCLGL




python /scratch1/mengxiwu/PCFEA/train_PCFEA_cls.py --dataroot /scratch1/mengxiwu/PCFEA/data/PointDA/ --src_dataset modelnet --trgt_dataset shapenet --seed 1 --model_type pcfea
