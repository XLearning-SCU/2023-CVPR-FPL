import logging
import argparse
import numpy as np
import os.path as osp
from collections import OrderedDict
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

from utils import util
from models import create_model
import options.options as option
from data import create_dataset, create_dataloader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to options YMAL file.')
    args = parser.parse_args()
    opt = option.dict_to_nonedict(option.parse(args.opt))

    util.mkdirs((path for key, path in opt['path'].items() \
        if not key=='pretrain_model' and not key=='root'))
    util.setup_logger(
        'base', opt['path']['log'], 'test',
        level=logging.INFO, screen=True, tofile=True
    )
    logger = logging.getLogger('base')
    logger.info(option.dict2str(opt))

    test_loaders = []
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, phase, dataset_opt)
        logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
        test_loaders.append(test_loader)

    model = create_model(opt)

    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.set_name
        logger.info('Testing [{:s}]...'.format(test_set_name))
        dataset_dir = osp.join(opt['path']['results_root'], test_set_name)
        util.mkdir(dataset_dir)

        need_GT=True
        test_results = OrderedDict()
        test_results['psnr'] = []
        test_results['ssim'] = []
        test_results['psnr_y'] = []
        test_results['ssim_y'] = []

        hr_test = True if test_set_name in ['Test2K', 'Test4K', 'Test8K'] \
            or opt['network']['which_model']=='SwinIR' else False

        for data in test_loader:

            if hr_test:
                model.hr_test(data)
            else:
                model.lr_test(data)
            
            visuals = model.get_current_visuals(need_GT)
            sr_img = util.tensor2img(visuals['rlt'], np.uint8, (0, 1))
            img_name = osp.basename(data['LQ_path'][0])
            util.save_img(sr_img, osp.join(dataset_dir, img_name))

            if need_GT:
                gt_img = util.tensor2img(visuals['GT'], np.uint8, (0, 1))
                sr_img, gt_img = util.crop_border([sr_img, gt_img], opt['scale'])
                psnr = compare_psnr(sr_img, gt_img, data_range=255)
                ssim = compare_ssim(sr_img, gt_img, data_range=255, multichannel=True)
                test_results['psnr'].append(psnr)
                test_results['ssim'].append(ssim)

                sr_img_y = util.bgr2ycbcr(sr_img / 255., only_y=True)
                gt_img_y = util.bgr2ycbcr(gt_img / 255., only_y=True)
                psnr_y = compare_psnr(sr_img_y * 255, gt_img_y * 255, data_range=255)
                ssim_y = compare_ssim(sr_img_y * 255, gt_img_y * 255, data_range=255, multichannel=True)
                test_results['psnr_y'].append(psnr_y)
                test_results['ssim_y'].append(ssim_y)
                
                logger.info('{:20s} - PSNR: {:.4f} - SSIM:{:.6f} - PSNR_Y: {:.4f} - SSIM_Y:{:.6f}'.format(
                    img_name, psnr, ssim, psnr_y, ssim_y
                ))
            else:
                logger.info(img_name)
        
        if need_GT:
            ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
            ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
            ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
            ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
            logger.info('Average PSNR:{:.4f}, SSIM:{:.6f}, PSNR_Y:{:.4f} and SSIM_Y:{:.6f} for {}'.format(
                ave_psnr, ave_ssim, ave_psnr_y, ave_ssim_y, test_set_name
            ))

if __name__=='__main__':
    main()