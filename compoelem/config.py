import os

import yaml

dirname = os.path.dirname(__file__)
config_file = os.path.join(dirname, './config.yaml')
with open(config_file, 'r') as stream:
    config = yaml.safe_load(stream)

# import yaml
# # try:
# from importlib import resources as res
# # except ImportError:
# #     import importlib_resources as res

# with res.open_binary('compoelem', 'config.yaml') as fp:
#     config = yaml.load(fp, Loader=yaml.Loader)