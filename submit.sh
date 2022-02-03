#!/bin/sh
#
#SBATCH --job-name=test
#SBATCH --output="output.txt"
#SBATCH --error="error.txt" 
#
#SBATCH --ntasks=1
#SBATCH --time=10:00
#SBATCH --mem-per-cpu=8192
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load releases/2020b
module load TensorFlow/2.4.1-fosscuda-2020b

srun pip3 install --user --upgrade pip
srun pip3 install --user scikit-build
srun pip3 install --user cmake
srun pip3 install --user -r requirements.txt
srun python3 unet_running.py

