#!/bin/bash
#SBATCH --account=qcb_640
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:a100:1
#SBATCH --time=47:55:00 
#SBATCH --mem=128GB 
#SBATCH --exclusive 
##SBATCH --gpu-freq=high
#SBATCH --job-name="Prod"
#SBATCH --constraint=epyc-7513
#SBATCH --mail-type=all
date

echo "$(date): job $SLURM_JOBID starting on $SLURM_NODELIST"
nvidia-smi



# ===================
# Module load
# ===================
module purge
module load gcc/11.3.0
module load cudnn/8.4.0.27-11.6
module load cuda/11.6.2
module load conda/4.12.0
module load git
conda init bash
conda config --set auto_activate_base false
eval "$(conda shell.bash hook)"
conda deactivate


# =============
# Prody
# ===============
conda activate /home1/homingla/.conda/envs/Prody23
which python


nohup /home1/homingla/.conda/envs/Prody23/bin/python  commandBenchmark_Prody_Linux_Pdb_0064_EPYC.py >/dev/null >8&1
nohup /home1/homingla/.conda/envs/Prody23/bin/python  commandBenchmark_ProdySparse_Linux_Pdb_0064_EPYC.py >/dev/null >9&1
