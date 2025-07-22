import sys
sys.path.insert(0, '.')
import logging
from torch.utils.data import DataLoader
from c3pi.dataloader import PPIDatasetFromFiles, protein_collate_fn
from c3pi.c3pi import FullPPIModel
from c3pi.train import train_model

import yaml
with open("configs/config_gold.yaml", 'r') as f:
    config = yaml.safe_load(f)

logging.basicConfig(
    filename=config['base']['log_dir']+config['base']['model_name']+'.txt',       # Log file path
    level=logging.INFO,            # Log level
    format='%(message)s'
)



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Dataset
    train_dataset = PPIDatasetFromFiles(
        config['base']['train_pair_dir'], config['base']['embedding_dir'])
    val_dataset = PPIDatasetFromFiles(
        config['base']['validation_pair_dir'], config['base']['embedding_dir'])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,
                              collate_fn=protein_collate_fn, num_workers=5)
    val_loader = DataLoader(val_dataset, batch_size=64,
                            collate_fn=protein_collate_fn, num_workers=5)

    # Model
    model = FullPPIModel(input_features_dim=1024)

    # Train
    train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=40,
        lr=1e-4,
        save_path=config['base']['checkpoint_dir'] + config['base']['model_name'],
        log_dir=config['base']['log_dir'] + config['base']['model_name'],
    )
