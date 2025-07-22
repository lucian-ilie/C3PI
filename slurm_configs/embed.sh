#!/bin/bash
#SBATCH --time=01:20:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --output=slurm_outputs/embed-%j.out  #%j for jobID
#SBATCH --mail-user=shosse59@uwo.ca
#SBATCH --mail-type=BEGIN,END,FAIL
export HF_HOME=HF_HOME/
python -u c3pi/embedder.py