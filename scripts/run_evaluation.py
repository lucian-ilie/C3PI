import sys
sys.path.insert(0, '.')

import yaml
import torch
from c3pi.utils import evaluate_models_varying_thresholds, average_predictions_by_group


with open("configs/config_gold.yaml", 'r') as f:
    config = yaml.safe_load(f)

output_path = config['base']['output_dir'] + config['base']['model_name'] + config['base']['organism'] + '.pt'

predictions = torch.load(output_path, map_location=torch.device('cpu')).squeeze().numpy()
average_predictions_by_group(input_csv=config['base']['test_pair_dir'],
                                            prediction_tensor_path=output_path,
                                            output_csv=config['base']['result_dir'] + config['base']['organism'] + '_averaged.tsv',
                                            group_size=8)


species_list = ['human']
model_list = ['fusion']
evaluate_models_varying_thresholds(species_list, model_list, results_dir='results', prediction_file=config['base']['result_dir'] + config['base']['organism'] + '_averaged.tsv')