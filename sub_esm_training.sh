#!/bin/bash
sbatch <<EOT
#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --mem=40G
#SBATCH --output=$PWD/slurm_out/output_${1}_${2}_${3}_%j.out
#SBATCH --job-name=${1}
###
source ~/.bash_profile
conda activate pytorch 

cd $PWD
python /n/home03/wayment/esm_BMRB_big/train.py --version $1 --missing_class_wt $2 --split $3
EOT

