#!/bin/sh
#
#SBATCH --job-name=test
#SBATCH --output=res.txt
#
#SBATCH --ntasks=1
#SBATCH --time=10:00
#SBATCH --mem-per-cpu=4096
#SBATCH --partition=gpu
#SBATCH --gres=gpu:GeForceRTX2080Ti:6

srun pip3 install --user --upgrade pip
srun pip3 install --user scikit-build
srun pip3 install --user cmake
srun pip3 install --user -r requirements.txt
srun python3 unet_running.py
