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
        