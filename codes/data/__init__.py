import torch
import logging
import torch.utils.data

def create_dataset(dataset_opt):
    mode = dataset_opt['mode']

    if mode == 'LQGT':
        from data.LQGT_dataset import LQGTDataset as D
    else:
        raise NotImplementedError(
            'Dataset [{:s}] is not recognized.'.format(mode)
        )
    dataset = D(dataset_opt)

    logging.getLogger('base').info(
        'dataset [{:s} - {:s}] is created.'.format(
            dataset.__class__.__name__, dataset_opt['name']
        )
    )
    return dataset

def create_dataloader(dataset, phase, dataset_opt, opt=None, sampler=None):
    if phase == 'train':
        if opt['dist']:
            world_size = torch.distributed.get_world_size()
            num_workers = dataset_opt['n_workers']
            assert dataset_opt['batch_size'] % world_size == 0
            batch_size = dataset_opt['batch_size'] // world_size
            shuffle = False
        else:
            num_workers = dataset_opt['n_workers'] * len(opt['gpu_ids'])
            batch_size = dataset_opt['batch_size']
            shuffle = True
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, sampler=sampler,
            drop_last=True, pin_memory=False
        )
    else:
        batch_size = dataset_opt['batch_size'] if dataset_opt['batch_size'] else 1
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=1, pin_memory=False
        )