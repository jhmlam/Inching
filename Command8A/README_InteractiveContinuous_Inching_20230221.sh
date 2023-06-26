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
# Inching
# ===============
conda activate /home1/homingla/.conda/envs/Inching23
which python
#nohup /home1/homingla/.conda/envs/Inching23/bin/python commandAnimate_pdb.py >/dev/null 2>&1
#nohup /home1/homingla/.conda/envs/Inching23/bin/python commandAnimate_cif.py >/dev/null 2>&1

#for i in {0..10}
#do
#/home1/homingla/.conda/envs/Inching23/bin/python commandBenchmark_Inching_Linux_Pdb_0064_A100.py
#nohup /home1/homingla/.conda/envs/Inching23/bin/python commandBenchmark_Inching_Linux_Pdb_0064_A100.py >/dev/null 2>&1
#nohup /home1/homingla/.conda/envs/Inching23/bin/python commandBenchmark_Inching_Linux_Cif_0064_A100.py >/dev/null 2>&1
#done


#for i in {0..10}
#do
#/home1/homingla/.conda/envs/Inching23/bin/python commandBenchmark_Inching_Linux_Pdb_0064_A100.py
#nohup /home1/homingla/.conda/envs/Inching23/bin/python commandBenchmark_Inching_Linux_Pdb_0064_P100.py >/dev/null 2>&1
#nohup /home1/homingla/.conda/envs/Inching23/bin/python commandBenchmark_Inching_Linux_Cif_0064_P100.py >/dev/null 2>&1
#done



# =========================
# Inching TRLM
# =========================
#for i in {0..10}
#do
#nohup /home1/homingla/.conda/envs/Inching23/bin/python commandBenchmark_InchingTRLM_Linux_Pdb_0064_A100.py >/dev/null 2>&1
#nohup /home1/homingla/.conda/envs/Inching23/bin/python commandBenchmark_InchingTRLM_Linux_Cif_0064_A100.py >/dev/null 2>&1
#nohup /home1/homingla/.conda/envs/Inching23/bin/python commandBenchmark_InchingTRLM_Linux_Pdb_0064_P100.py >/dev/null 2>&1
#nohup /home1/homingla/.conda/envs/Inching23/bin/python commandBenchmark_InchingTRLM_Linux_Cif_0064_P100.py >/dev/null 2>&1
#done


# =========================
# Inching JDM
# =========================
#for i in {0..10}
#do
#nohup /home1/homingla/.conda/envs/Inching23/bin/python commandBenchmark_InchingJDM_Linux_Pdb_0064_A100.py >/dev/null 2>&1
#nohup /home1/homingla/.conda/envs/Inching23/bin/python commandBenchmark_InchingJDM_Linux_Cif_0064_A100.py >/dev/null 2>&1
#nohup /home1/homingla/.conda/envs/Inching23/bin/python commandBenchmark_InchingJDM_Linux_Pdb_0064_P100.py >/dev/null 2>&1
#nohup /home1/homingla/.conda/envs/Inching23/bin/python commandBenchmark_InchingJDM_Linux_Cif_0064_P100.py >/dev/null 2>&1
#/home1/homingla/.conda/envs/Inching23/bin/python commandBenchmark_InchingJDM_Linux_Cif_0064_P100.py
#done




# =========================
# Inching JDM SHowcase
# =========================
#for i in {0..10}
#do
#nohup /home1/homingla/.conda/envs/Inching23/bin/python commandBenchmark_InchingJDM_Linux_CifShowcase_0064_A100.py >/dev/null 2>&1
#nohup /home1/homingla/.conda/envs/Inching23/bin/python commandBenchmark_InchingJDM_Linux_CifShowcase_0064_P100.py >/dev/null 2>&1
#done

# =========================
# Inching JDM Showcase on NPC with periodic boundary
# =========================
for i in {0..10}
do
nohup /home1/homingla/.conda/envs/Inching23/bin/python commandBenchmark_InchingJDM_Linux_CifShowcase_0064_A100_Npc.py >/dev/null 2>&1
done


# ==========================
# 
