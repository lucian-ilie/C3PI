# C3PI: Protein-Protein Interaction Prediction

This repository contains code for **C3PI**, a deep learning framework designed for protein-protein interaction (PPI) prediction using protein sequence embeddings. The project integrates advanced embedding models, multi-scale convolutional neural networks, and robust evaluation pipelines to predict interactions between proteins.

---

## Repository Structure
```
├── c3pi
│ ├── c3pi.py # Core model definition (FullPPIModel and building blocks)
│ ├── dataloader.py # Dataset classes and data loading utilities
│ ├── embedder.py # Sequence embedding generation using ProtTransT5XLU50Embedder
│ ├── init.py
│ ├── predict.py # Prediction utilities (not detailed here)
│ ├── train.py # Training and validation functions
│ └── utils.py # Evaluation metrics, averaging, and embedding cleanup utilities
├── configs
│   └── config_gold.yaml  # YAML conf. file with paths and parameters. You must configure this file first, as the default addresses are placeholders.
├── data
│ ├── permutations.npy # Precomputed permutations used for puzzler
│ └── permute.py # Script to generate diverse permutations with max Hamming distance
├── scripts # This directory contains everything you need to run the model, unless you plan to modify its core logic.
│ ├── run_embedder.py # Script to generate embeddings from FASTA sequences based on config
│ ├── run_evaluation.py # Script to average predictions and evaluate model performance
│ ├── run_prediction.py # Script for running model inference
│ └── run_training.py # Script to train the FullPPIModel using dataset and config
├── slurm_configs
│ ├── embed.sh # SLURM batch script for embedding generation
│ ├── predict.sh # SLURM batch script for prediction
│ └── train.sh # SLURM batch script for training
└── README.md # This file
```


---

## Setup and Requirements

- Python 3.8+
- PyTorch
- Bio-embeddings (`bio-embeddings` package)
- scikit-learn
- PyYAML
- TensorBoard (optional, for training logs)

Install dependencies with:

```bash
pip install torch bio-embeddings scikit-learn pyyaml tensorboard
```

## Usage
1. Generate Protein Sequence Embeddings
``` bash

python scripts/run_embedder.py
```
This will read sequences from the FASTA file specified in fasta_path and generate embeddings saved in embedding_dir. Existing embeddings are skipped to save time.

2. Train the PPI Model (optional) 
``` bash
python scripts/run_training.py
```
Trains the FullPPIModel using training and validation pairs from the configured dataset. Training logs are saved in the directory set by `log_dir`.

3. Run Predictions (Inference)
```bash

python scripts/run_prediction.py
```
(This script runs the trained model on test data and saves the raw predictions. To use the trained model, you need to download the weights by following the instructions in `checkpoints/README.md`.
)

4. Evaluate Model Performance
```bash
python scripts/run_evaluation.py
```
This script averages predictions in groups (default group size: 8), saves averaged predictions, and computes metrics (accuracy, precision, recall, AUROC, AUPR, F1, MCC) across multiple thresholds.

