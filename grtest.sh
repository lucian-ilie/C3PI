#!/bin/bash
#SBATCH --time=11:58:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=90G
#SBATCH --gres=gpu:t4:1       
#SBATCH --output=c3pi-%j.out  #%j for jobID

source /path/to/mainENV


python predict.py
