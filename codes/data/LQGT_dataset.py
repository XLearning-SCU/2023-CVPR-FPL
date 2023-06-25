import torch
import numpy as np
import torch.utils.data as data

from utils import util

class LQGTDataset(data.Dataset):
    def __init__(self, opt):
        super().__init__()

        self.set_name = opt['name']
        self.use_flip = opt['use_flip']
        self.use_rot = opt['use_rot']
        self.paths_GT = util.get_image_paths(opt['data_type'], opt['dataroot_GT'])
        self.paths_LQ = util.get_image_paths(opt['data_type'], opt['dataroot_LQ'])
        
        assert len(self.paths_LQ) == len(self.paths_GT), \
            'GT and LQ datasets have different number of images - {}, {}.'.format(
                len(self.paths_LQ), len(self.paths_GT)
            )

    def __getitem__(self, index):
        GT_path = self.paths_GT[index]
        LQ_path = self.paths_LQ[index]

        img_GT = util.read_img(GT_path)
        img_LQ = util.read_img(LQ_path)

        img_GT, img_LQ = util.augment(
            [img_GT, img_LQ], self.use_flip, self.use_rot
        )

        if img_GT.shape[2] == 3: # BGR to RGB
            img_GT = img_GT[:, :, [2, 1, 0]]
            img_LQ = img_LQ[:, :, [2, 1, 0]]
        
        img_GT = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))
        ).float()
        img_LQ = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))
        ).float()

        return {'LQ_path': LQ_path, 'GT_path': GT_path, 'LQ': img_LQ, 'GT': img_GT}

    def __len__(self):
        return len(self.paths_GT)