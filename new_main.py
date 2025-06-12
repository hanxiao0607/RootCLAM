import json

with open('new_config.json', 'r') as f:
    cfg = json.load(f)
print('=' * 50,'\nConfig:')
for key, value in cfg.items():
    print(f'    {key}: {value}')
# print('Configuration loaded:', cfg)
print('=' * 50)

