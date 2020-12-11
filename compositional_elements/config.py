import os

import yaml

dirname = os.path.dirname(__file__)
config_file = os.path.join(dirname, './config.yaml')
with open(config_file, 'r') as stream:
    config = yaml.safe_load(stream)
