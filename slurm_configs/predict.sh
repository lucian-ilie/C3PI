#!/bin/bash
#SBATCH --time=01:50:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --output=slurm_outputs/predict-%j.out  #%j for jobID
#SBATCH --mail-user=shosse59@uwo.ca
#SBATCH --mail-type=BEGIN,END,FAIL
source /home/mohsenh/projects/def-ilie/mohsenh/ENV/prostt5ENV/bin/activate
python -u scripts/run_prediction.py