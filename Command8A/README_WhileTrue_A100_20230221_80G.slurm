#!/bin/bash
#SBATCH --account=qcb_640
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:a100:1
#SBATCH --time=47:55:00 
#SBATCH --mem=128GB 
#SBATCH --exclusive 
##SBATCH --gpu-freq=high
#SBATCH --job-name="a100G"
#SBATCH --constraint=epyc-7513,a100-80gb


#SBATCH --mail-type=all

module purge
module load gcc/11.3.0
module load cudnn/8.4.0.27-11.6
module load cuda/11.6.2
module load conda/4.12.0
module load git
conda init bash
conda config --set auto_activate_base false
eval "$(conda shell.bash hook)"


# https://www.carc.usc.edu/user-information/user-guides/software-and-programming/anaconda

conda config --set auto_activate_base false
conda deactivate


date

echo "$(date): job $SLURM_JOBID starting on $SLURM_NODELIST"
nvidia-smi



# ===================
# Module load
# ===================


# =============
# Inching
# ===============
conda activate /home1/homingla/.conda/envs/Inching23
which python

for i_PdbCif in Pdb Cif CifShowcase
do


    for i_machine in A100
    do

        # ==========================
        # Inching TRLM
        # ==========================
        sed "s#BenchmarkLinuxCupy0064#BenchmarkLinuxInchingTRLM0064_${i_machine}#g" ./Template/commandBenchmark_InchingTRLM_Linux_${i_PdbCif}.py > commandBenchmark_InchingTRLM_Linux_${i_PdbCif}_0064_${i_machine}.py
        sed -i 's#User_n_mode = 64#User_n_mode = 64#g' commandBenchmark_InchingTRLM_Linux_${i_PdbCif}_0064_${i_machine}.py
        mkdir ../BenchmarkLinuxInchingTRLM0064_${i_machine}


        sed "s#BenchmarkLinuxCupy0064#BenchmarkLinuxInchingTRLMHD0064_${i_machine}#g" ./Template/commandBenchmark_InchingTRLMHD_Linux_${i_PdbCif}.py > commandBenchmark_InchingTRLMHD_Linux_${i_PdbCif}_0064_${i_machine}.py
        sed -i 's#User_n_mode = 64#User_n_mode = 64#g' commandBenchmark_InchingTRLMHD_Linux_${i_PdbCif}_0064_${i_machine}.py
        mkdir ../BenchmarkLinuxInchingTRLMHD0064_${i_machine}

        sed "s#BenchmarkLinuxCupy0064#BenchmarkLinuxInchingTRLMFull0064_${i_machine}#g" ./Template/commandBenchmark_InchingTRLMFull_Linux_${i_PdbCif}.py > commandBenchmark_InchingTRLMFull_Linux_${i_PdbCif}_0064_${i_machine}.py
        sed -i 's#User_n_mode = 64#User_n_mode = 64#g' commandBenchmark_InchingTRLMFull_Linux_${i_PdbCif}_0064_${i_machine}.py
        mkdir ../BenchmarkLinuxInchingTRLMFull0064_${i_machine}


         
        # ================================================
        # Inching JDM
        # ================================================
        sed "s#BenchmarkLinuxCupy0064#BenchmarkLinuxInchingJDM0064_${i_machine}#g" ./Template/commandBenchmark_InchingJDM_Linux_${i_PdbCif}.py > commandBenchmark_InchingJDM_Linux_${i_PdbCif}_0064_${i_machine}.py
        sed -i 's#User_n_mode = 64#User_n_mode = 64#g' commandBenchmark_InchingJDM_Linux_${i_PdbCif}_0064_${i_machine}.py
        mkdir ../BenchmarkLinuxInchingJDM0064_${i_machine}



        sed "s#BenchmarkLinuxCupy0064#BenchmarkLinuxInchingJDMHD0064_${i_machine}#g" ./Template/commandBenchmark_InchingJDMHD_Linux_${i_PdbCif}.py > commandBenchmark_InchingJDMHD_Linux_${i_PdbCif}_0064_${i_machine}.py
        sed -i 's#User_n_mode = 64#User_n_mode = 64#g' commandBenchmark_InchingJDMHD_Linux_${i_PdbCif}_0064_${i_machine}.py
        mkdir ../BenchmarkLinuxInchingJDMHD0064_${i_machine}

        sed "s#BenchmarkLinuxCupy0064#BenchmarkLinuxInchingJDMFull0064_${i_machine}#g" ./Template/commandBenchmark_InchingJDMFull_Linux_${i_PdbCif}.py > commandBenchmark_InchingJDMFull_Linux_${i_PdbCif}_0064_${i_machine}.py
        sed -i 's#User_n_mode = 64#User_n_mode = 64#g' commandBenchmark_InchingJDMFull_Linux_${i_PdbCif}_0064_${i_machine}.py
        mkdir ../BenchmarkLinuxInchingJDMFull0064_${i_machine}

        # ================================================
        # Inching IRLM
        # ================================================
        sed "s#BenchmarkLinuxCupy0064#BenchmarkLinuxInchingIRLM0064_${i_machine}#g" ./Template/commandBenchmark_InchingIRLM_Linux_${i_PdbCif}.py > commandBenchmark_InchingIRLM_Linux_${i_PdbCif}_0064_${i_machine}.py
        sed -i 's#User_n_mode = 64#User_n_mode = 64#g' commandBenchmark_InchingIRLM_Linux_${i_PdbCif}_0064_${i_machine}.py
        mkdir ../BenchmarkLinuxInchingIRLM0064_${i_machine}



        sed "s#BenchmarkLinuxCupy0064#BenchmarkLinuxInchingIRLMHD0064_${i_machine}#g" ./Template/commandBenchmark_InchingIRLMHD_Linux_${i_PdbCif}.py > commandBenchmark_InchingIRLMHD_Linux_${i_PdbCif}_0064_${i_machine}.py
        sed -i 's#User_n_mode = 64#User_n_mode = 64#g' commandBenchmark_InchingIRLMHD_Linux_${i_PdbCif}_0064_${i_machine}.py
        mkdir ../BenchmarkLinuxInchingIRLMHD0064_${i_machine}

        sed "s#BenchmarkLinuxCupy0064#BenchmarkLinuxInchingIRLMFull0064_${i_machine}#g" ./Template/commandBenchmark_InchingIRLMFull_Linux_${i_PdbCif}.py > commandBenchmark_InchingIRLMFull_Linux_${i_PdbCif}_0064_${i_machine}.py
        sed -i 's#User_n_mode = 64#User_n_mode = 64#g' commandBenchmark_InchingIRLMFull_Linux_${i_PdbCif}_0064_${i_machine}.py
        mkdir ../BenchmarkLinuxInchingIRLMFull0064_${i_machine}


        # ==============================
        # Inching Hessian and ARPACK
        # ===============================
        j_machine='EPYC'
        sed "s#BenchmarkLinuxArpack0064#BenchmarkLinuxArpack0064_${j_machine}#g" ./Template/commandArpack_${i_PdbCif}_0064.py > commandArpack_${i_PdbCif}_0064_${j_machine}.py
        sed -i 's#User_n_mode = 64#User_n_mode = 64#g' commandArpack_${i_PdbCif}_0064_${j_machine}.py
        mkdir ../BenchmarkLinuxArpack0064_${j_machine}

        # ==============================
        # Prody
        # =============================
        mkdir ../BenchmarkLinuxPrody0064_${j_machine}
        sed "s#BenchmarkLinuxPrody0064#BenchmarkLinuxPrody0064_${j_machine}#g" ./Template/commandBenchmark_Prody_Linux_${i_PdbCif}_0064.py > commandBenchmark_Prody_Linux_${i_PdbCif}_0064_${j_machine}.py

        mkdir ../BenchmarkLinuxProdySparse0064_${j_machine}
        sed "s#BenchmarkLinuxPrody0064#BenchmarkLinuxProdySparse0064_${j_machine}#g" ./Template/commandBenchmark_ProdySparse_Linux_${i_PdbCif}_0064.py > commandBenchmark_ProdySparse_Linux_${i_PdbCif}_0064_${j_machine}.py



    done
done






# =========================
# Halt
# =============================



x=1
while [ $x -le 5 ]
do
  x=$(( $x ))
done



# 0064
#/home1/homingla/.conda/envs/Inching11/bin/python  commandBenchmark_Inching_Linux_Cif_0064_A100.py
# nohup /home1/homingla/.conda/envs/Inching11/bin/python commandArpack_Cif_0064_A100.py >/dev/null 2>&1

