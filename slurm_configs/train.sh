#!/bin/bash
#SBATCH --time=20:10:00
#SBATCH --account=def-ilie
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100
#SBATCH --cpus-per-task=6
#SBATCH --mem=70G
#SBATCH --output=slurm_outputs/train-%j.out  #%j for jobID
#SBATCH --mail-user=shosse59@uwo.ca
#SBATCH --mail-type=BEGIN,END,FAIL
source /home/mohsenh/projects/def-ilie/mohsenh/ENV/prostt5ENV/bin/activate
python -u scripts/run_training.py