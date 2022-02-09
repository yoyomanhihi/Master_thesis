#!/bin/sh
#
#SBATCH --job-name=name
#SBATCH --output="nodataaugm.txt" 
#
#SBATCH --ntasks=1
#SBATCH --time=36:00:00
#SBATCH --mem-per-cpu=8192
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load releases/2019b
module load TensorFlow/2.3.1-fosscuda-2019b-Python-3.7.4

srun pip3 install --user --upgrade pip 
srun pip3 install --user scikit-build
srun pip3 install --user cmake 
srun pip3 install --user -r requirements.txt
srun pip3 install --user tensorflow
srun nvidia-smi
srun python3 -c 'from tensorflow.python.client import device_lib;print(device_lib.list_local_devices())'
srun python3 main.py nodataaugm
