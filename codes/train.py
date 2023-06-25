import torch
import os, math
import argparse
import numpy as np
import random, logging
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

from utils import util
from models import create_model
import options.options as option
from data import create_dataloader, create_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument('-DDP', action='store_true', default=False, help='DDP if true else DP')
    args = parser.parse_args()
    opt = option.dict_to_nonedict(option.parse(args.opt))

    if args.DDP:
        opt['dist'] = True
        if mp.get_start_method(allow_none=True) != 'spawn':
            mp.set_start_method('spawn')
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        dist.init_process_group(backend='nccl')
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        if rank == 0:
            print('enable distributed data parallel training')
    else:
        opt['dist'], rank = False, -1
        print('enable data parallel training')

    if rank <= 0:
        util.mkdir_and_rename(opt['path']['experiments_root'])
        util.mkdirs((path for key, path in opt['path'].items() \
            if 'root' not in key and key!='pretrain_model' and key!='resume_state' and key!='strict_load'))
        util.setup_logger('base', opt['path']['log'], 'train', level=logging.INFO, screen=True, tofile=True)
        logger = logging.getLogger('base')
        logger.info(option.dict2str(opt))
        if opt['use_tb_logger']:
            tb_logger = SummaryWriter(log_dir='../tb_logger/'+opt['name'])
    else:
        util.setup_logger('base', opt['path']['log'], 'train', level=logging.INFO, screen=True)
        logger = logging.getLogger('base')

    if opt['path']['resume_state']:
        resume_state = torch.load(
            opt['path']['resume_state'],
            map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device())
        )
        opt['path']['pretrain_model'] = os.path.join(opt['path']['models'],'{}.pth'.format(resume_state['iter']))
        if rank <= 0:
            logger.warning('pretrain_model path will be ignored when resuming training.')
            logger.info('set pretrain_model to ' + opt['path']['pretrain_model'])
    else:
        resume_state = None
    
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    util.set_random_seed(seed)
    if rank <= 0:
        logger.info('random seed: {}'.format(seed))

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set)/dataset_opt['batch_size']))
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters/train_size))
            train_sampler = DistributedSampler(train_set, world_size, rank) if opt['dist'] else None
            train_loader = create_dataloader(train_set, phase, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info('number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
                logger.info('total epochs needed: {:d} for iters {:,d}'.format(total_epochs, total_iters))
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, phase, dataset_opt, opt, None)
            if rank <= 0:
                logger.info('number of val images in [{:s}]: {:d}'.format(dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError('phase [{:s}] is not recognized.'.format(phase))
    
    model = create_model(opt)

    if resume_state:
        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)
        if rank <= 0:
            logger.info('Resuming training from epoch: {}, iter: {}.'.format(start_epoch, current_step))
    else:
        current_step, start_epoch = 0, 0

    for epoch in range(start_epoch, total_epochs+1):
        if opt['dist']:
            train_sampler.set_epoch(epoch)

        for train_data in train_loader:
            current_step += 1
            if current_step > total_iters:
                break

            model.train(train_data)

            if current_step % opt['train']['print_freq'] == 0:
                logs = model.get_current_log()
                message = '[rank:{:2d}, epoch:{:3d}, iter:{:8d}, lr:({:.3e})]'.format(
                    rank, epoch, current_step, model.get_current_learning_rate()
                )
                for k, v in logs.items():
                    message += ' {:s}: {:.4e} '.format(k, v)
                    if rank <= 0 and opt['use_tb_logger']:
                        tb_logger.add_scalar(k, v, current_step)
                logger.info(message)
            
            if rank <= 0 and opt['datasets']['val'] \
                and current_step % opt['train']['val_freq'] == 0:
                
                pbar = util.ProgressBar(len(val_loader))
                avg_psnr, avg_ssim, idx = 0., 0., 0

                hr_test = True if val_loader.dataset.set_name in ['Test2K', 'Test4K', 'Test8K'] \
                        or opt['network']['which_model']=='SwinIR' else False
                
                for val_data in val_loader:
                    idx += 1

                    model.hr_test(val_data) if hr_test else model.lr_test(val_data)
                    
                    visuals = model.get_current_visuals()
                    sr_img = util.tensor2img(visuals['rlt'], np.uint8, (0, 1))
                    gt_img = util.tensor2img(visuals['GT'], np.uint8, (0, 1))

                    img_name = os.path.splitext(os.path.basename(val_data['LQ_path'][0]))[0]
                    if opt['datasets']['val']['save_results']:
                        img_dir = os.path.join(opt['path']['val_images'], img_name)
                        util.mkdir(img_dir)
                        save_img_path = os.path.join(img_dir,'{:s}_{:d}.png'.format(img_name, current_step))
                        util.save_img(sr_img, save_img_path)

                    sr_img, gt_img = util.crop_border([sr_img, gt_img], opt['scale'])
                    avg_psnr += compare_psnr(sr_img, gt_img, data_range=255)
                    avg_ssim += compare_ssim(sr_img, gt_img, data_range=255, multichannel=True)

                    pbar.update('Test {}'.format(img_name))
                
                avg_psnr = avg_psnr/idx
                avg_ssim = avg_ssim/idx
                
                logger.info('validation PSNR: {:.4e}, SSIM: {:.6e}'.format(avg_psnr, avg_ssim))
                if opt['use_tb_logger']:
                    tb_logger.add_scalar('psnr', avg_psnr, current_step)
                    tb_logger.add_scalar('ssim', avg_ssim, current_step)
                        
            if rank <= 0 and \
                current_step % opt['train']['save_freq'] == 0:
                logger.info('saving models and training states')
                model.save(current_step, opt['path']['models'])
                model.save_training_state(epoch, current_step, opt['path']['training_state'])
    
    if rank <= 0:
        logger.info('end of the training.')
        tb_logger.close()

if __name__ == '__main__':
    main()