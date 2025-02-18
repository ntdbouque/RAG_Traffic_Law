'''
Author: Nguyen Truong Duy
Purpose: Create a YamlParser class for read yaml file config and merge yaml file config
Latest Update: 06/02/2025
'''

import os
import yaml
from easydict import EasyDict as edict

class YamlParser(edict):
    '''
    This is yaml parser based on EasyDict
    '''

    def __init__(self, cfg_dict=None, config_file=None):
        if cfg_dict is None:
            cfg_dict = {}

        if config_file is not None:
            assert(os.path.isfile(config_file))
            with open(config_file, 'r') as fo:
                cfg_dict.update(yaml.load(fo.read(), Loader=yaml.FullLoader))

        super(YamlParser, self).__init__(cfg_dict)
        
    def merge_from_file(self, config_file):
        with open(config_file, 'r') as fo:
            self.update(yaml.load(fo.read(), Loader=yaml.FullLoader))

    def merge_from_dict(self, config_dict):
        self.update(config_dict)

def get_config(config_file=None):
    return YamlParser(config_file=config_file)

if __name__ == '__main__':
    config_file = r'C:\Users\duy\Desktop\TrafficLaw\config\config.yaml'
    cfg = get_config(config_file)
    print(cfg)