from easydict import EasyDict
from pathlib import Path
import yaml


def log_config_to_file(cfg, pre='cfg', logger=None):
    for key, val in cfg.items():
        if isinstance(cfg[key], EasyDict):
            logger.info('\n%s.%s = edict()' % (pre, key))
            log_config_to_file(cfg[key], pre=pre + '.' + key, logger=logger)
            continue
        logger.info('%s.%s: %s' % (pre, key, val))


def cfg_from_list(cfg_list, config):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = config
        for subkey in key_list[:-1]:
            assert subkey in d, 'NotFoundKey: %s' % subkey
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d, 'NotFoundKey: %s' % subkey
        try:
            value = literal_eval(v)
        except:
            value = v

        if type(value) != type(d[subkey]) and isinstance(d[subkey], EasyDict):
            key_val_list = value.split(',')
            for src in key_val_list:
                cur_key, cur_val = src.split(':')
                val_type = type(d[subkey][cur_key])
                cur_val = val_type(cur_val)
                d[subkey][cur_key] = cur_val
        elif type(value) != type(d[subkey]) and isinstance(d[subkey], list):
            val_list = value.split(',')
            for k, x in enumerate(val_list):
                val_list[k] = type(d[subkey][0])(x)
            d[subkey] = val_list
        else:
            assert type(value) == type(d[subkey]), \
                'type {} does not match original type {}'.format(type(value), type(d[subkey]))
            d[subkey] = value


def cfg_from_yaml_file(cfg_file, config):
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.load(f)
        config.update(EasyDict(new_config))

    cfg_preprocess(cfg)
    return config


def cfg_preprocess(cfg):
    cfg.TORCH_VOXEL_GENERATOR = cfg.USE_PSEUDOLIDAR or cfg.INJECT_SEMANTICS


cfg = EasyDict()
cfg.ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
cfg.LOCAL_RANK = 0

# default flags
cfg.TAG_PTS_WITH_RGB = False
cfg.MODE = '3dobjdet'
cfg.ALTERNATE_PT_CLOUD_ABS_DIR = ''  # default to empty str which represents False
cfg.PERCENT_OF_PTS = 100  # by default, use all of the points
cfg.TAG_PTS_IF_IN_GT_BBOXES = False

cfg.INJECT_SEMANTICS = False
cfg.INJECT_SEMANTICS_HEIGHT = 0
cfg.INJECT_SEMANTICS_WIDTH = 0
cfg.INJECT_SEMANTICS_MODE = 'binary_car_mask'  # one of binary_car_mask, logit_car_mask

cfg.TRAIN_SEMANTIC_NETWORK = False
cfg.SEMANTICS_ZERO_OUT = False
cfg.USE_PSEUDOLIDAR = False

# cfg.TORCH_VOXEL_GENERATOR  # set by a combination of other scripts
