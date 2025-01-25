#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=2GB
#SBATCH --time=00:5:00
#SBATCH --output=pcfea.out


sbatch pcfea/ms1.sh
sbatch pcfea/ms2.sh
sbatch pcfea/ms3.sh
sbatch pcfea/s+s1.sh
sbatch pcfea/s+s2.sh
sbatch pcfea/s+s3.sh
sbatch pcfea/ss+1.sh
sbatch pcfea/ss+2.sh
sbatch pcfea/ss+3.sh
sbatch pcfea/sm1.sh
sbatch pcfea/sm2.sh
sbatch pcfea/sm3.sh
sbatch pcfea/s+m1.sh
sbatch pcfea/s+m2.sh
sbatch pcfea/s+m3.sh
sbatch pcfea/ms+1.sh
sbatch pcfea/ms+2.sh
sbatch pcfea/ms+3.sh
