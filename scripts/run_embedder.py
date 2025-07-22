import sys
sys.path.insert(0, '.')

import yaml
import os
from c3pi.embedder import process_sequences

def main():
    # Load config
    with open("configs/config_gold.yaml", 'r') as f:
        config = yaml.safe_load(f)

    fasta_path = config['base']['fasta_path']
    output_dir = config['base']['embedding_dir']

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating embeddings from {fasta_path} to {output_dir} ...")
    process_sequences(fasta_path, output_dir)
    print("Embedding generation completed.")

if __name__ == "__main__":
    main()
