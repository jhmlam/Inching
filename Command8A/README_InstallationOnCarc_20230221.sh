
# ===================
#  Inching23
# ======================
#module purge
#module load gcc/11.2.0
#module load cudnn/8.2.4.15-11.4
#module load cuda/11.4.0

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
#conda create --name Inching23 python=3.8.12
conda activate Inching23

# NOTE Torch passed! You need to specify pytorch=1.11.0=py3.8_cuda11.3_cudnn8.2.0_0
#conda install -y -c conda-forge -c pytorch scipy=1.7.1 pytorch=1.11.0=py3.8_cuda11.3_cudnn8.2.0_0 cudatoolkit=11.3 seaborn=0.11.2 plotly=5.3.1 prody=2.0 pandas=1.3.3 mdtraj=1.9.6 openmm=7.6.0 tqdm numba

#conda install -y -c conda-forge -c pytorch scipy=1.7.1 pytorch=1.11.0=py3.8_cuda11.3_cudnn8.2.0_0 cudatoolkit=11.3 seaborn=0.11.2 plotly=5.3.1 prody=2.0 pandas=1.3.3 mdtraj=1.9.6 openmm=7.6.0 tqdm numba cutensor=1.5.0.3 cupy=11.1.0

# NOTE torch.zeros(3,device =0) failed
#conda install -y -c conda-forge -c pytorch scipy=1.8.0 scikit-learn=1.0.2 pytorch=1.13.1 cudatoolkit=11.6 seaborn=0.11.2 plotly=5.3.1 pandas=1.3.3 mdtraj=1.9.6 openmm=7.6.0 tqdm numba cutensor=1.6.2.3 cupy=11.5.0


# NOTE does not even import torcjh
#conda install -y -c conda-forge -c pytorch -c nvidia scipy=1.8.0 scikit-learn=1.0.2 pytorch-cuda=11.6 seaborn=0.11.2 plotly=5.3.1 pandas=1.3.3 mdtraj=1.9.6 openmm=7.6.0 tqdm numba cutensor=1.6.2.3 cupy=11.5.0

# NOTE does notwork
#conda install -y -c conda-forge -c pytorch -c nvidia scipy=1.8.0 scikit-learn=1.0.2 pytorch=1.13.1=py3.8_cuda11.6 seaborn=0.11.2 plotly=5.3.1 pandas=1.3.3 mdtraj=1.9.6 openmm=7.6.0 tqdm numba cutensor=1.6.2.3 cupy=11.5.0

# NOTE No cuda from torch
#conda install -y -c conda-forge -c pytorch -c nvidia scipy=1.8.0 scikit-learn=1.0.2 pytorch==1.12.1 cudatoolkit=11.6 seaborn=0.11.2 plotly=5.3.1 pandas=1.3.3 mdtraj=1.9.6 openmm=7.6.0 tqdm numba cutensor=1.6.2.3 cupy=11.5.0

# NOTE
#conda install -y -c conda-forge -c pytorch scipy=1.8.0 pytorch=1.11.0=py3.8_cuda11.3_cudnn8.2.0_0 cudatoolkit=11.3 seaborn=0.11.2 plotly=5.3.1 prody=2.0 pandas=1.3.3 mdtraj=1.9.6 openmm=7.6.0 tqdm numba cutensor=1.6.2.3 cupy=11.5.0


# NOTE Works? seems yes
conda install -y -c conda-forge -c pytorch scipy=1.8.0 pytorch=1.11.0=py3.8_cuda11.3_cudnn8.2.0_0 cudatoolkit=11.3 seaborn=0.11.2 plotly=5.3.1  pandas=1.3.3 mdtraj=1.9.6 openmm=7.6.0 tqdm numba cutensor=1.6.2.3 cupy=11.5.0

salloc --partition=debug --gres=gpu:p100:1
conda activate Inching23



/home1/homingla/.conda/envs/Inching23/bin/python

conda deactivate 

# ==================================
# Install pymol
# ===================================


module purge
module load gcc/11.2.0
module load cudnn/8.2.4.15-11.4
module load cuda/11.4.0
module load anaconda3
module load git
conda config --set auto_activate_base false
conda create --name PymolOnly python=3.8.12
conda activate PymolOnly


conda install -c schrodinger pymol-bundle=2.5.4
conda isntall ffmpeg




conda deactivate 

# =============================
# Getting Prody
# =========================



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
conda create --name Prody23 python=3.8.12
conda activate Prody23

# NOTE Works? seems yes
conda install -y -c conda-forge -c pytorch scipy=1.8.0 pytorch=1.11.0=py3.8_cuda11.3_cudnn8.2.0_0 cudatoolkit=11.3 seaborn=0.11.2 plotly=5.3.1  pandas=1.3.3 mdtraj=1.9.6 openmm=7.6.0 tqdm numba cutensor=1.6.2.3 cupy=11.5.0 prody=2.4


conda deactivate 