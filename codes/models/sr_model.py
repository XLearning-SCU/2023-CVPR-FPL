import os
import torch
import logging
import functools
import torch.nn as nn
from thop import profile
import torch.nn.functional as F
from collections import OrderedDict
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel

from utils import util
from models import networks

logger = logging.getLogger('base')

class SRModel():
    def __init__(self, opt):
        super().__init__()
        
        train_opt = opt['train']
        self.test_scale = opt['scale']
        self.test_batch = 32 # opt['test']['batch']
        self.test_size = 32 # opt['test']['size']
        self.test_step = 28 # opt['test']['step']
        self.rank = torch.distributed.get_rank() \
            if opt['dist'] else -1
        
        self.net = networks.define(opt)
        # self.print_network()
        self.net = self.net.cuda()
        self.net = DistributedDataParallel(self.net) \
            if opt['dist'] else DataParallel(self.net)
        self.load(opt)

        self.upscale_func = functools.partial(
            F.interpolate, mode='bicubic', align_corners=False
        ) if train_opt['loss'] == 'fpl' else None

        if opt['is_train']:
            self.net.train()
            
            if train_opt['loss'] == 'l1':
                logger.info('enable l1 loss for training')
                self.cri_pix = nn.L1Loss(reduction='mean').cuda()
            elif train_opt['loss'] == 'l2':
                logger.info('enable l2 loss for training')
                self.cri_pix = nn.MSELoss(reduction='mean').cuda()
            elif train_opt['loss'] == 'fpl':
                logger.info('enable focal pixel learning')
                self.cri_pix = util.focal_pixel_learning().cuda()
            self.cri_typ = train_opt['loss']

            self.optimizer = torch.optim.Adam(
                self.net.parameters(), lr=train_opt['lr']
            )
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, train_opt['T_period'], train_opt['eta_min']
            )
            self.log_dict = OrderedDict()

    def train(self, data):
        self.var_L = data['LQ'].cuda()
        self.real_H = data['GT'].cuda()
        self.fake_H = self.net(self.var_L)

        l_pix = self.cri_pix(self.fake_H, self.real_H, self.var_L) \
            if self.cri_typ == 'fpl' else self.cri_pix(self.fake_H, self.real_H)
        
        self.optimizer.zero_grad()
        l_pix.backward()
        self.optimizer.step()
        self.scheduler.step()

        self.log_dict['l_pix'] = l_pix.item()

    def hr_test(self, data):
        self.net.eval()
        self.var_L = data['LQ'].cuda()
        self.real_H = data['GT'].cuda()
        lr_h, lr_w = self.var_L.size()[2:]
        lr_list = util.image2patch(
            self.var_L, self.test_size, self.test_step
        )
        with torch.no_grad():
            if self.upscale_func is None:
                sr = torch.cat([
                    self.net(lr) \
                        for lr in torch.split(lr_list, self.test_batch, dim=0)
                ], dim=0)
            else:
                sr = torch.cat([
                    self.net(lr) \
                        + self.upscale_func(lr, scale_factor=self.test_scale) \
                            for lr in torch.split(lr_list, self.test_batch, dim=0)
                ], dim=0)
        self.fake_H = util.patch2image(
            sr, lr_h*self.test_scale, lr_w*self.test_scale,
            self.test_step*self.test_scale
        )
        self.net.train()
    
    def lr_test(self, data):
        self.net.eval()
        self.var_L = data['LQ'].cuda()
        self.real_H = data['GT'].cuda()
        with torch.no_grad():
            self.fake_H = self.net(self.var_L)
            if self.upscale_func is not None:
                self.fake_H = self.fake_H + self.upscale_func(
                    self.var_L, scale_factor=self.test_scale
                )
        self.net.train()

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach().squeeze(0).float().cpu()
        out_dict['rlt'] = self.fake_H.detach().squeeze(0).float().cpu()
        if need_GT:
            out_dict['GT'] = self.real_H.detach().squeeze(0).float().cpu()
        return out_dict

    def get_current_log(self):
        return self.log_dict
    
    def get_current_learning_rate(self):
        curr_lr = [param_group['lr'] for param_group in self.optimizer.param_groups]
        return curr_lr[0]

    def load(self, opt):
        load_path = opt['path']['pretrain_model']
        if load_path is not None:
            logger.info('Loading model for [{:s}] ...'.format(load_path))
            load_net = torch.load(load_path)
            if isinstance(self.net, nn.DataParallel) or isinstance(self.net, DistributedDataParallel):
                network = self.net.module
            load_net_clean = OrderedDict()
            for k, v in load_net.items():
                if k.startswith('module.'):
                    load_net_clean[k[7:]] = v
                else:
                    load_net_clean[k] = v
            network.load_state_dict(load_net_clean, strict=opt['path']['strict_load'])
        
    def save(self, iter_label, path):
        save_filename = '{}.pth'.format(iter_label)
        save_path = os.path.join(path, save_filename)
        if isinstance(self.net, nn.DataParallel) or isinstance(self.net, DistributedDataParallel):
            network = self.net.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def save_training_state(self, epoch, iter_step, path):
        state = {
            'epoch': epoch, 'iter': iter_step, 
            'scheduler': self.scheduler.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        save_path = os.path.join(path, '{}.state'.format(iter_step))
        torch.save(state, save_path)

    def resume_training(self, resume_state):
        self.optimizer.load_state_dict(resume_state['optimizer'])
        self.scheduler.load_state_dict(resume_state['schedulers'])

    def print_network(self):
        if self.rank <= 0:
            if isinstance(self.net, DataParallel) or \
                isinstance(self.net, DistributedDataParallel):
                network = self.net.module
                net_struc_str = '{} - {}'.format(
                    self.net.__class__.__name__, network.__class__.__name__
                )
            else:
                network = self.net
                net_struc_str = '{}'.format(network.__class__.__name__)
            macs, params = profile(
                network, inputs=(torch.randn(1,3,self.test_size,self.test_size),), verbose=False
            )
            logger.info('Network G structure: {}, with parameters: {:,d} and flops: {:,d}'.format(
                net_struc_str, int(params), int(2*macs)
            ))