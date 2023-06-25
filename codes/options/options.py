import os
import yaml
import os.path as osp
from utils.util import OrderedYaml
Loader, Dumper = OrderedYaml()

def parse(opt_path):
    with open(opt_path, mode='r') as f:
        opt = yaml.load(f, Loader=Loader)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = opt['gpu_ids']

    opt['path']['root'] = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir, osp.pardir))
    if opt['is_train']:
        experiments_root = osp.join(opt['path']['root'], 'experiments', opt['name'])
        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = osp.join(experiments_root, 'models')
        opt['path']['training_state'] = osp.join(experiments_root, 'training_state')
        opt['path']['val_images'] = osp.join(experiments_root, 'val_images')
        opt['path']['log'] = experiments_root
    else:
        results_root = osp.join(opt['path']['root'], 'results', opt['name'])
        opt['path']['results_root'] = results_root
        opt['path']['log'] = results_root
    
    return opt

class NoneDict(dict):
    def __missing__(self, key):
        return None

def dict_to_nonedict(opt):
    '''convert to NoneDict, which return None for missing key.'''
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt

def dict2str(opt, indent_l=1):
    '''dict to string for logger'''
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' '*(indent_l * 2)+k+':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' '*(indent_l * 2)+']\n'
        else:
            msg += ' '*(indent_l * 2)+k+': '+str(v)+'\n'
    return msg