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
python /n/home03/wayment/software/esm/train_hkws.py --version $1 --finetuning_method $2 --split $3
EOT

