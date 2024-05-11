#!/bin/bash
#SBATCH --time=100:10:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=90G
#SBATCH --output=c3pi-%j.out  #%j for jobID

source /path/to/mainENV

python trainCNN1D.py
python trainCNN2D.py