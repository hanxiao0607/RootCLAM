import json
from utils.constants import Cte


print('=' * 50,'\nConfig:')
with open('new_config.json', 'r') as f:
    cfg = json.load(f)
for key, value in cfg.items():
    print(f'    {key}: {value}')

print('=' * 50,'\nArgument:')

with open('new_argument.json', 'r') as f:
    args = json.load(f)
for key, value in args.items():
    print(f'    {key}: {value}')
print('=' * 50)

######################################################
dataset_name = 'adult'

if cfg['dataset']['name'] in Cte.DATASET_LIST:
    from data_modules.het_scm import HeterogeneousSCMDataModule

    dataset_params = cfg['dataset']['params'].copy()
    dataset_params['dataset_name'] = cfg['dataset']['name']
    if dataset_name == 'donors':
        dataset_params['num_samples_tr'] = args.training_size
    elif dataset_name in ['adult', 'loan']:
        dataset_params['num_samples_tr'] = args.training_size * 10
    else:
        NotImplementedError

    data_module = HeterogeneousSCMDataModule(**dataset_params)

    data_module.prepare_data()

print(data_module)