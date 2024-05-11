#!/bin/bash
#SBATCH --time=02:50:00
#SBATCH --cpus-per-task=3
#SBATCH --mem=30G
#SBATCH --output=c3pi-%j.out  #%j for joID

source /path/to/embedENV

python embedCreatorT5Functional.py   
