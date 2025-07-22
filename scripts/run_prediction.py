import sys
sys.path.insert(0, '.')

import torch
import logging
import yaml
from torch.utils.data import DataLoader
from c3pi.dataloader import PPIDatasetFromFiles, protein_collate_fn
from c3pi.predict import predict
from c3pi.c3pi import FullPPIModel

def main() -> None:
    # Load config
    with open("configs/config_gold.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Set up logging
    log_path = config['base']['log_dir'] + config['base']['model_name'] + '_predict.txt'
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(message)s'
    )

    logging.info("Starting prediction...")

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load model
    model = FullPPIModel(input_features_dim=1024).to(device)

    # Initialize lazy layers (if any)
    dummy_input = torch.randn(2, 795, 1024).to(device)
    model(dummy_input, dummy_input)
    logging.info("Model initialized.")

    # Load checkpoint
    checkpoint_path = config['base']['checkpoint_dir'] + config['base']['model_name']
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    logging.info(f"Loaded checkpoint from {checkpoint_path}")

    # Load dataset
    test_dataset = PPIDatasetFromFiles(
        config['base']['test_pair_dir'],
        config['base']['embedding_dir']
    )
    test_loader: DataLoader = DataLoader(
        test_dataset,
        batch_size=64,
        collate_fn=protein_collate_fn,
        num_workers=5
    )

    # Predict
    predictions = predict(model, test_loader, device)

    # Save predictions
    output_path = config['base']['output_dir'] + config['base']['model_name'] + config['base']['organism'] + '.pt'
    torch.save(predictions, output_path)
    logging.info(f"Saved predictions to {output_path}")

if __name__ == '__main__':
    main()
