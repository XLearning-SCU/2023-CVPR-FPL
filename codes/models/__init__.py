import logging

def create_model(opt):
    task = opt['task']
    
    if task == 'sr': 
        from .sr_model import SRModel as M
    else:
        raise NotImplementedError('task [{:s}] not recognized'.format(task))
    m = M(opt)

    logging.getLogger('base').info(
        'model [{:s}] is created'.format(m.__class__.__name__)
    )
    return m